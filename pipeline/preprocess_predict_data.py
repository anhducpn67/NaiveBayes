import project_config as config
import utility


class PreprocessPredictData:
    def __init__(self, container):
        self.container = container
        self.predict_data = self.container.TEST_POS_EXAMPLES

    def load_predict_data(self):
        self.predict_data = list(open(config.PREDICT_DATA_PATH, "r", errors='ignore').readlines())
        self.predict_data = [s.strip() for s in self.predict_data]

    def preprocess_sentence(self):
        self.predict_data = [utility.clean_str(sent) for sent in self.predict_data]

    def vectorizer_sentence(self):
        self.container.PREDICT_DATA = self.container.VECTORIZER.transform(self.predict_data).toarray()

    def execute(self):
        self.load_predict_data()
        self.preprocess_sentence()
        self.vectorizer_sentence()
        return self.container
