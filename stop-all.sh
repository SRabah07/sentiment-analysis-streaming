echo "Stop all started things..."

echo "Stop Kafka..."

echo "Stop Kafka Zookeeper..."
/opt/kafka_2.13-2.8.0/bin/zookeeper-server-stop.sh /opt/kafka_2.13-2.8.0/config/zookeeper.properties &
sleep 5

echo "Stop Kafka Server..."
/opt/kafka_2.13-2.8.0/bin/kafka-server-stop.sh /opt/kafka_2.13-2.8.0/config/server.properties &
sleep 5