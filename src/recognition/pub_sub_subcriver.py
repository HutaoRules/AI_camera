from google.cloud import pubsub_v1
import json
from stgcnpp_infer import run_inference

def callback(message):
    data = json.loads(message.data.decode("utf-8"))
    run_inference(data)
    message.ack()

def main():
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path("camera-ai-458010", "stgcn-sub")
    subscriber.subscribe(sub_path, callback)
    print("Listening for messages...")
    import time
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()