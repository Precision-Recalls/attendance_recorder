import json
import logging
from confluent_kafka import Consumer, KafkaException, KafkaError
from src.utils.configUtils import load_config

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_consumer(config):
    conf = config['kafka_consumer']['conf']
    return Consumer(**conf)


def process_message(msg):
    try:
        message_value = msg.value().decode('utf-8')
        message_key = msg.key().decode('utf-8') if msg.key() else None
        logger.info(f'Received message: {message_value} (key: {message_key})')

        # Deserialize JSON message
        message_data = json.loads(message_value)
        logger.info(f'Processed message data: {message_data}')
    except json.JSONDecodeError:
        logger.error('Failed to decode JSON message')


def consume_messages(consumer, topic_name):
    consumer.subscribe([topic_name])
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError.PARTITION_EOF:
                    logger.info(f'{msg.topic()} [{msg.partition()}] reached end at offset {msg.offset()}')
                else:
                    raise KafkaException(msg.error())
            else:
                process_message(msg)
    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    finally:
        consumer.close()


def main():
    try:
        config = load_config('config.ini')
        logger.info("Configuration loaded successfully.")

        topic_name = config['kafka']['topic_name']
        consumer = create_consumer(config)

        consume_messages(consumer, topic_name)
    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
