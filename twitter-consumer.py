#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Python permettant de récupérer en temps réel (streaming) des messages kafka qui correspondent au messages twitter portant sur la vaccination Covid19
en France. 

Note: Ce script utilise les librairies suivantes :

pyspark
kafka
pandas
numpy
tensorflow
datatime.

Le but est d'analyser le sentiment décrit par le message portant sur la vaccination en deux catégories : positif/négatif.
A cet effet nous utilisons un modèle entrainé auparavant sur un dataset de message twitter disponible Kaggle (https://www.kaggle.com/hbaflast/french-twitter-sentiment-analysis)
en langue Française. 

Une fois le text du commentaire évalué il sera rendu disponible pour un access en temps réel par Tableau et sauvegarder dans Mongo.
"""

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[PID#%(process)d]:%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("twitter-consumer.log"),
        logging.StreamHandler()
    ]
)

# Pyspark part
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StringType, StructField 
import pyspark.sql.functions as F

from py4j.java_gateway import java_import


# Basic Libraries
import pandas as pd
import numpy as np

# Text Processing utisant NTLK
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re,string,unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, LSTM, GRU, GlobalMaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Bidirectional
import os

# Load the stored Tokenizer 
import pickle
 
# Mongo
from pymongo import MongoClient
MONGO_DB_URL = '192.168.204.146'
#MONGO_DB_URL = os.environ['IP_MONGODB']
MONGO_DB_PORT = 27018

class MongoDBUtils:
    """
    Mongo DB Utils class
    '''

    Attributes
    ----------

    Methods
    -------
    get_db():
        Get the scrapping Mongo DB .
    """
    CLIENT = MongoClient(MONGO_DB_URL, MONGO_DB_PORT)
    
    @staticmethod
    def get_db():
        return MongoDBUtils.CLIENT.twitter


mongo_db= MongoDBUtils.get_db()

# Pre Processing du text 

# nlk french stopword
nltk.download('stopwords')
stop = set(stopwords.words('french'))

# Les méthodes ci dessous sont les même que celle utilisé lors de la mise en place de l'IA
# de l'analyse des sentiment des tweets en Français.

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

def remove_hashtag(text):
    return re.sub(r'@.\w*', '', text)
    
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

# Clean the text 
def clean_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    text = remove_hashtag(text)
    return text

# Load the Tokenizer
tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

if not tokenizer:
    raise ValueError('Tokenizer cannot be None.')

# Load the model
logging.info('Loading tensorflow model...')
model = tf.keras.models.load_model('./LSTM')
logging.info('Loading tensorflow model done.')

def predict(df, _):
     df = df.toPandas()
     size = df.shape[0]
     if size > 0:
       logging.info('Cleaning the twitter comment...')
       df['cleaned_text']=df['text'].apply(clean_text)
       logging.info(df[['cleaned_text']].head())

       logging.info('Tokenizing the twitter comment...')
       tokenized_corpus = tokenizer.texts_to_sequences(df['text'])
       logging.info('Pad sequence...')
       data = pad_sequences(tokenized_corpus, padding='post', maxlen = 280)
       logging.info(data)

       logging.info('Start Prediction...')
       predictions = model.predict(data) 
       logging.info('End Prediction...')
       logging.info(predictions)

       sentiments = []
       predictions_as_list=[]
       sentiments_as_list=[]
       index = 0
       for item in df.itertuples():
        prediction = float(predictions[index][0])
        sentiment_value='Positif' if  prediction > 0.5 else 'Négatif'
        sentiment = {
           'date': item.created_at,
           'text': item.text,
           'cleaned_text': item.cleaned_text,
           'prediction': prediction,
           'sentiment': sentiment_value,
           'payload': item.payload
        }
        
        sentiments.append(sentiment)
        predictions_as_list.append(prediction)
        sentiments_as_list.append(sentiment_value)
        index+=1

       # Insert into MongoDB
       mongo_db.sentiments.insert_many(sentiments)

       # Convert to Spark DF and insert into Hive
       df['prediction']=predictions_as_list
       df['sentiment']=sentiments_as_list

       sparkDF=spark.createDataFrame(df)
       sparkDF.write.mode('append').saveAsTable("default.Twitter")

spark = SparkSession \
    .builder \
    .appName("Project-Final") \
    .enableHiveSupport() \
    .config('spark.sql.hive.thriftServer.singleSession', True) \
    .getOrCreate()

sc = spark.sparkContext 


java_import(sc._gateway.jvm,"")

#Start the Thrift Server using the jvm and passing the same spark session corresponding to pyspark session 
# in the jvm side.
sc._gateway.jvm.org.apache.spark.sql.hive.thriftserver.HiveThriftServer2.startWithContext(spark._jwrapped)

logging.info('Start Building the Schema ....')
schema = StructType() \
      .add('created_at',StringType()) \
      .add('text', StringType()) \
      .add('payload', StringType())

logging.info('Build the Spark streaming pipeline to read from Kafka...')
inputData = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "twitter-topic") \
  .load() \
  .select(from_json(F.col("value").cast('string'), schema).alias("value")) \
  .selectExpr("value.created_at", "value.text", "value.payload")

logging.info('Start processing the kafka message by batch ...')
query = inputData \
    .writeStream \
    .foreachBatch(predict) \
    .start()

query.awaitTermination()
logging.info('End Processing the kafka message by batch ...')