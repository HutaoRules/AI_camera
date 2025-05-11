from kafka import KafkaConsumer
from cassandra.cluster import Cluster
import json

cluster = Cluster(['127.0.0.1'])
session = cluster.connect()
session.execute("""
CREATE KEYSPACE IF NOT EXISTS surveillance
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
""")
session.set_keyspace('surveillance')

session.execute("""
CREATE TABLE IF NOT EXISTS pose_data (
    id UUID PRIMARY KEY,
    timestamp DOUBLE,
    keypoints TEXT,
    bbox TEXT
)
""")

from uuid import uuid4

consumer = KafkaConsumer('pose-keypoints', bootstrap_servers='localhost:9092')

for msg in consumer:
    data = json.loads(msg.value)
    session.execute("""
        INSERT INTO pose_data (id, timestamp, keypoints, bbox)
        VALUES (%s, %s, %s, %s)
    """, (uuid4(), data['timestamp'], json.dumps(data['keypoints']), json.dumps(data['bbox'])))
