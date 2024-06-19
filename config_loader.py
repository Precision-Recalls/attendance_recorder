import configparser


def load_config(config_file):
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    except Exception as e:
        print(f"Error occurred in configuration loading: {e}")
