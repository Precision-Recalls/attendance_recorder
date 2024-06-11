from confluent_kafka import Consumer, KafkaException, KafkaError
import json

# Configuration for the Kafka consumer
conf = {
    'bootstrap.servers': 'localhost:9092',  # Replace with your Kafka broker(s)
    'group.id': 'python-consumer-group',
    'auto.offset.reset': 'earliest'
}

# Create a Kafka consumer
consumer = Consumer(**conf)
consumer.subscribe(['test-topic'])

try:
    while True:
        # Poll for new messages
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError.PARTITION_EOF:
                # End of partition event
                print(f'{msg.topic()} [{msg.partition()}] reached end at offset {msg.offset()}')
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            # Successfully received a message
            message_value = msg.value().decode('utf-8')
            message_key = msg.key().decode('utf-8') if msg.key() else None

            # Process the message
            print(f'Received message: {message_value} (key: {message_key})')

            # If your messages are JSON encoded, you can deserialize them
            try:
                message_data = json.loads(message_value)
                print(f'Processed message data: {message_data}')
            except json.JSONDecodeError:
                print('Failed to decode JSON message')

except KeyboardInterrupt:
    print('Interrupted by user')
finally:
    # Close down consumer to commit final offsets
    consumer.close()

