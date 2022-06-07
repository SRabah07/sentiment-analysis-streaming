#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Python permettant de récupérer en temps réel (streaming) des messages twitter portant sur la vaccination Covid19
en France. Une fois récupérer ils seront publier sur un Broker Kafka.

Note: Ce script utilise les librairies suivantes :

tweepy
kafka
os
json
datatime.

Note: Pour utiliser la librairie tweepy (https://www.tweepy.org/), vous aurez besoin de configuer un compte twitter 
pour développeur (https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api).
Ce script aura besoin des informations suivantes qui doivent être en variables d'environement

- API Key -> TWITTER_API_KEY
- API Key Secret -> TWITTER_API_SECRET
- Access Token -> TWITTER_ACCESS_TOKEN
- Access Token Secret -> TWITTER_ACCESS_SECRET
"""

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[PID#%(process)d]:%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("twitter-producer.log"),
        logging.StreamHandler()
    ]
)


# Kafka
from kafka import KafkaProducer

# Tweepy
import tweepy
import os
from datetime import datetime
import json

# Authentification Twitter
API_KEY = os.environ['TWITTER_API_KEY']
API_SECRET = os.environ['TWITTER_API_SECRET']
ACCESS_TOKEN = os.environ['TWITTER_ACCESS_TOKEN']
ACCESS_SECRET = os.environ['TWITTER_ACCESS_SECRET']

# Create the Kafka Producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# Streaming Listener
class TwitterSentimentListener(tweepy.StreamListener):
    """
    Custom Stream Listener that inherent from Tweepy base class and implements the logic 
    of getting twitter message and publish them into kafka broker.
    """

    def on_status(self, status):
        try:
            """
            Gets a new twitter status payload: 
            """
            logging.info('Getting new status!')
            text=''
            if hasattr(status, "retweeted_status"):  # Check if Retweet
                try:
                    text=status.retweeted_status.extended_tweet["full_text"]
                except AttributeError:
                    text=status.retweeted_status.text
            else:
                try:
                    text=status.extended_tweet["full_text"]
                except AttributeError:
                    text=status.text

            data = {
                'created_at': status.created_at.isoformat(),
                'text': text,
                'payload': status._json
            }
            logging.info('--------------------------------------')
            logging.info(f'{data}')
            logging.info('--------------------------------------')
            producer.send('twitter-topic', data)
        except Exception as e:
            logging.error(f'Error occurs {e}')

    def on_error(self, status_code):
        logging.error(f'Error occurs {status_code}')


def main():
    """
    Retrieves the Twitter messages and publish them to Kafka. 
    """
    listener = TwitterSentimentListener() 
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET) 
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True)
    twitter_stream = tweepy.Stream(api.auth, listener)
    twitter_stream.filter(track=['vaccin covid'], languages = ['fr'])



if __name__ == "__main__":
    main()
