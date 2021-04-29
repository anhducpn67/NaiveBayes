import pickle

import project_config as config


class CacheData:
    def __init__(self, container):
        self.container = container
        self.is_used_cache = config.IS_USED_CACHE_PREPROCESSED_DATA

    def load_data_from_cache(self):
        # Load data
        self.container.TRAIN_POS_DATA = pickle.load(open(config.TRAINING_POS_DATA_PATH, 'rb'))
        self.container.TRAIN_NEG_DATA = pickle.load(open(config.TRAINING_NEG_DATA_PATH, 'rb'))
        # Load params
        params = pickle.load(open(config.PARAMS_CACHE_PATH, 'rb'))
        self.container.WORDS_POS_DATA = params[0]
        self.container.WORDS_NEG_DATA = params[1]
        self.container.LEN_VOCAB = params[2]
        # Load vectorizer
        self.container.VECTORIZER = pickle.load(open(config.VECTORIZER_CACHE_PATH, 'rb'))
        # Load count-vectors
        self.container.COUNT_VECTOR_POS, \
            self.container.COUNT_VECTOR_NEG = pickle.load(open(config.COUNT_VECTOR_CACHE_PATH, 'rb'))

    def cache_to_file(self):
        # Dump training data and test data
        pickle.dump(self.container.TRAIN_POS_DATA, open(config.TRAINING_POS_DATA_PATH, 'wb'))
        pickle.dump(self.container.TRAIN_NEG_DATA, open(config.TRAINING_NEG_DATA_PATH, 'wb'))
        # Dump params
        params = [self.container.WORDS_POS_DATA,
                  self.container.WORDS_NEG_DATA,
                  self.container.LEN_VOCAB]
        pickle.dump(params, open(config.PARAMS_CACHE_PATH, 'wb'))
        # Dump vectorizer
        pickle.dump(self.container.VECTORIZER, open(config.VECTORIZER_CACHE_PATH, 'wb'))
        # Dump count-vectors
        pickle.dump([self.container.COUNT_VECTOR_POS, self.container.COUNT_VECTOR_NEG],
                    open(config.COUNT_VECTOR_CACHE_PATH, 'wb'))

    def execute(self):
        if self.is_used_cache:
            self.load_data_from_cache()
        else:
            self.cache_to_file()
        return self.container
