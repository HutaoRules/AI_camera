from google.cloud import pubsub_v1
import json
import time

# Cấu hình
project_id = "camera-ai-458010"
topic_id = "keypoint-topic"
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

# Dữ liệu mẫu
def create_sample_keypoint_data():
    return {
        "id": 1,
        "keypoints": [[100, 200], [150, 250], [120, 210],'dotuan']
    }

def main():
    for i in range(5):
        data = create_sample_keypoint_data()
        data["id"] = i
        data_str = json.dumps(data)
        future = publisher.publish(topic_path, data=data_str.encode("utf-8"))
        print(f"Sent: {data}")
        time.sleep(1)

if __name__ == "__main__":
    main()


from google.cloud import pubsub_v1
import json

project_id = "camera-ai-458010"
subscription_id = "keypoint-sub"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

response = subscriber.pull(subscription=subscription_path, max_messages=5)

for msg in response.received_messages:
    print(f"Received: {msg.message.data.decode('utf-8')}")
    subscriber.acknowledge(subscription=subscription_path, ack_ids=[msg.ack_id])
