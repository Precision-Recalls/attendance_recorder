from confluent_kafka import Producer
import time
import json
import random


# Load the configuration
config = load_config('config.ini')
conf = config['kafka_consumer']['conf']
topic_name = config['kafka']['topic_name']

# Create a Kafka producer
producer = Producer(**conf)


def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')


try:
    while True:
        # Generate a random message
        message = {
            'timestamp': time.time(),
            'value': random.randint(0, 100)
        }

        # Convert the message to a JSON string
        message_str = json.dumps(message)

        # Produce the message to the Kafka topic
        producer.produce(topic_name, key=str(message['timestamp']), value=message_str, callback=delivery_report)

        # Poll to handle delivery reports (callbacks)
        producer.poll(0)

        # Sleep for a while before sending the next message
        time.sleep(1)
except KeyboardInterrupt:
    print('Interrupted by user')
finally:
    # Wait for any outstanding messages to be delivered and delivery reports to be received
    producer.flush()
