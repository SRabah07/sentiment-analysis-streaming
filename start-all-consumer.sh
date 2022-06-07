echo "Mongo DB will be started at Port  27018..."
echo "Getting IP address of WSL"
IP=$(hostname -I)

echo "IP Address is $IP"

export IP_MONGODB=$IP

echo "Start Spark Part..."

echo "Start Spark Server"
/opt/spark-3.1.2/sbin/start-master.sh &
sleep 5

echo "Start Spark Worker"
/opt/spark-3.1.2/sbin/start-worker.sh spark://[HOST]:7077 --cores 1 &
sleep 5

echo "Start Spark Consumer"
spark-submit --master spark://[HOST]:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 twitter-consumer.py &