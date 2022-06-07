echo "Start all needed things..."

echo "Kafka..."

echo "Start Kafka Zookeeper..."
/opt/kafka_2.13-2.8.0/bin/zookeeper-server-start.sh /opt/kafka_2.13-2.8.0/config/zookeeper.properties &
sleep 5

echo "Start Kafka Server..."
/opt/kafka_2.13-2.8.0/bin/kafka-server-start.sh /opt/kafka_2.13-2.8.0/config/server.properties &
sleep 5

echo "Create Kafka Topic..."
/opt/kafka_2.13-2.8.0/bin/kafka-topics.sh --create --topic twitter-topic --bootstrap-server localhost:9092 &
sleep 5


echo "Mongo DB will be started at Port  27018..."
echo "Getting IP address of WSL"
IP=$(hostname -I)

echo "IP Address is $IP"

export IP_MONGODB=$IP
mongod --port=27018 --bind_ip=$IP &
mongo --port=27018 --host=$IP &

echo "Export Needed Things"

export TWITTER_API_KEY=XXXX
export TWITTER_API_SECRET=XXXX
export TWITTER_ACCESS_TOKEN=XXXX
export TWITTER_ACCESS_SECRET=XXXX
export TWITTER_TOKEN=XXXX


echo "Start Producer : Twitter Streaming"
python twitter-producer.py &
sleep 5

echo "Start Spark Part..."

echo "Start Spark Server"
/opt/spark-3.1.2/sbin/start-master.sh &
sleep 5

echo "Start Spark Worker"
/opt/spark-3.1.2/sbin/start-worker.sh spark://DESKTOP-RTHD0A9.localdomain:7077 --cores 1 &
sleep 5

echo "Start Spark Consumer"
spark-submit --master spark://DESKTOP-RTHD0A9.localdomain:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 twitter-consumer.py &