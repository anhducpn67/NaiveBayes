from typing import Any


class Container:
    """
    Used to carrying object through out the pipeline
    """

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __init__(self):
        self.VOCAB = list()
        self.TRAIN_POS_EXAMPLES = list()
        self.TRAIN_NEG_EXAMPLES = list()
        self.TEST_POS_EXAMPLES = list()
        self.TEST_NEG_EXAMPLES = list()
        self.TOTAL_EXAMPLES = None
        self.VECTORIZER = None

        self.TRAIN_POS_DATA = list()
        self.TRAIN_NEG_DATA = list()
        self.TEST_POS_DATA = list()
        self.TEST_NEG_DATA = list()
        self.PREDICT_DATA = list()

        self.LEN_TRAIN_POS_DATA = None
        self.LEN_TRAIN_NEG_DATA = None
        self.WORDS_POS_DATA = None
        self.WORDS_NEG_DATA = None
        self.LEN_VOCAB = None

        self.COUNT_VECTOR_POS = None
        self.COUNT_VECTOR_NEG = None
