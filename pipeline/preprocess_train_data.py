from sklearn.feature_extraction.text import CountVectorizer

import project_config as config
import utility


class PreprocessTrainData:
    def __init__(self, container):
        self.container = container
        self.is_used_cache = config.IS_USED_CACHE_PREPROCESSED_DATA
        self.train_pos_example = self.container.TRAIN_POS_EXAMPLES
        self.train_neg_example = self.container.TRAIN_NEG_EXAMPLES

    def preprocess_sentence(self):
        self.train_pos_example = [utility.clean_str(sent) for sent in self.train_pos_example]
        self.train_neg_example = [utility.clean_str(sent) for sent in self.train_neg_example]

    def vectorizer_sentence(self):
        self.container.VECTORIZER = CountVectorizer()
        bag_of_words = self.container.VECTORIZER.fit_transform(self.train_pos_example + self.train_neg_example).toarray()
        self.container.LEN_VOCAB = bag_of_words.shape[1]
        self.container.TRAIN_POS_DATA = bag_of_words[:self.container.LEN_TRAIN_POS_DATA]
        self.container.TRAIN_NEG_DATA = bag_of_words[self.container.LEN_TRAIN_POS_DATA:]

    def calc_count_vector(self):
        self.container.COUNT_VECTOR_POS = [1] * self.container.LEN_VOCAB
        self.container.COUNT_VECTOR_NEG = [1] * self.container.LEN_VOCAB
        for vector in self.container.TRAIN_POS_DATA:
            self.container.COUNT_VECTOR_POS += vector
        for vector in self.container.TRAIN_NEG_DATA:
            self.container.COUNT_VECTOR_NEG += vector
        self.container.WORDS_POS_DATA = sum(self.container.COUNT_VECTOR_POS)
        self.container.WORDS_NEG_DATA = sum(self.container.COUNT_VECTOR_NEG)

    def execute(self):
        if self.is_used_cache:
            return self.container
        else:
            self.preprocess_sentence()
            self.vectorizer_sentence()
            self.calc_count_vector()
        return self.container
