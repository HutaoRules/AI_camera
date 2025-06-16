from cassandra.cluster import Cluster
from cassandra.query import dict_factory

def get_cassandra_session():
    cluster = Cluster(["127.0.0.1"])  # IP của Cassandra
    session = cluster.connect("smartcam")  # Tên keyspace
    session.row_factory = dict_factory
    return session

def insert_video_metadata(session, camera_id, filepath, timestamp):
    session.execute("""
        INSERT INTO video_metadata (camera_id, timestamp, filepath)
        VALUES (%s, %s, %s)
    """, (camera_id, timestamp, filepath))
