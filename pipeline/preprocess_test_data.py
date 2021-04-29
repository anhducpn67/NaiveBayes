import utility


class PreprocessTestData:
    def __init__(self, container):
        self.container = container
        self.test_pos_example = self.container.TEST_POS_EXAMPLES
        self.test_neg_example = self.container.TEST_NEG_EXAMPLES

    def preprocess_sentence(self):
        self.test_pos_example = [utility.clean_str(sent) for sent in self.test_pos_example]
        self.test_neg_example = [utility.clean_str(sent) for sent in self.test_neg_example]

    def vectorizer_sentence(self):
        self.container.TEST_POS_DATA = self.container.VECTORIZER.transform(self.test_pos_example).toarray()
        self.container.TEST_NEG_DATA = self.container.VECTORIZER.transform(self.test_neg_example).toarray()

    def execute(self):
        self.preprocess_sentence()
        self.vectorizer_sentence()
        return self.container
