import project_config


class Model:
    def __init__(self, container):
        self.container = container
        self.training_pos_examples = container.TRAIN_POS_DATA
        self.training_neg_examples = container.TRAIN_NEG_DATA
        self.test_pos_examples = container.TEST_POS_DATA
        self.test_neg_examples = container.TEST_NEG_DATA

    @staticmethod
    def calc_prob(sentence, numbers_class, numbers_examples, total_words, count_vector):
        prob = numbers_class / numbers_examples
        for idx, element in enumerate(sentence):
            temp = count_vector[idx] / total_words
            for i in range(0, element):
                prob = prob * temp
        return prob

    def classify_sent(self, sentence):
        prob_positive = Model.calc_prob(sentence,
                                        self.container.LEN_TRAIN_POS_DATA,
                                        self.container.LEN_TRAIN_POS_DATA + self.container.LEN_TRAIN_NEG_DATA,
                                        self.container.WORDS_POS_DATA,
                                        self.container.COUNT_VECTOR_POS)
        prob_negative = Model.calc_prob(sentence,
                                        self.container.LEN_TRAIN_NEG_DATA,
                                        self.container.LEN_TRAIN_POS_DATA + self.container.LEN_TRAIN_NEG_DATA,
                                        self.container.WORDS_NEG_DATA,
                                        self.container.COUNT_VECTOR_NEG)
        if prob_positive > prob_negative:
            return 1
        return 0

    def predict(self):
        for sent in self.container.PREDICT_DATA:
            prediction = self.classify_sent(sent)
            if prediction == 0:
                print("Negative review")
            else:
                print("Positive review")

    def evaluate_model(self):
        numbers_true_pos_predicts = 0
        numbers_true_neg_predicts = 0
        numbers_test_data = project_config.NUMBERS_TEST_DATA_EACH_CLASS * 2
        for sent in self.test_pos_examples:
            predict = self.classify_sent(sent)
            if predict == 1:
                numbers_true_pos_predicts += 1
        for sent in self.test_neg_examples:
            predict = self.classify_sent(sent)
            if predict == 0:
                numbers_true_neg_predicts += 1
        print("True positive:", numbers_true_pos_predicts)
        print("True negative:", numbers_true_neg_predicts)
        print("Score:", ((numbers_true_neg_predicts + numbers_true_pos_predicts) / numbers_test_data) * 100)

    def execute(self):
        return self.container
