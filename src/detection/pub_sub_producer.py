from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("camera-ai-458010", "stgcn-keypoints")

def send_keypoints(data):
    json_data = json.dumps(data).encode("utf-8")
    future = publisher.publish(topic_path, json_data)
    future.result()  # wait for publish (hoặc dùng async)
