import os


class Configuration:
    PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
    CACHE_PATH = PROJECT_PATH + '\\cache'
    IMAGE_PATH = PROJECT_PATH + '\\dataset\\images'
    DATA_ENTRY_PATH = PROJECT_PATH + '\\dataset\\data_entry_2017.csv'
