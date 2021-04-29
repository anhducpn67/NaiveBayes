import project_config as config


class LoadData:
    def __init__(self, container):
        self.container = container
        self.positive_examples = None
        self.negative_examples = None

    def load_data_and_labels(self):
        self.positive_examples = list(open(config.RAW_POS_DATA_PATH, "r", errors='ignore').readlines())
        self.positive_examples = [s.strip() for s in self.positive_examples]
        self.negative_examples = list(open(config.RAW_NEG_DATA_PATH, "r", errors='ignore').readlines())
        self.negative_examples = [s.strip() for s in self.negative_examples]

    def split_data(self):
        self.container.TOTAL_EXAMPLES = len(self.positive_examples) + len(self.negative_examples)
        self.container.TRAIN_POS_EXAMPLES = self.positive_examples[:-config.NUMBERS_TEST_DATA_EACH_CLASS]
        self.container.TRAIN_NEG_EXAMPLES = self.negative_examples[:-config.NUMBERS_TEST_DATA_EACH_CLASS]
        self.container.TEST_POS_EXAMPLES = self.positive_examples[-config.NUMBERS_TEST_DATA_EACH_CLASS:]
        self.container.TEST_NEG_EXAMPLES = self.negative_examples[-config.NUMBERS_TEST_DATA_EACH_CLASS:]
        # Params
        self.container.LEN_TRAIN_POS_DATA = len(self.container.TRAIN_POS_EXAMPLES)
        self.container.LEN_TRAIN_NEG_DATA = len(self.container.TRAIN_NEG_EXAMPLES)

    def execute(self):
        self.load_data_and_labels()
        self.split_data()
        return self.container
