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
        true_positives = 0
        true_negatives = 0
        numbers_test_data = project_config.NUMBERS_TEST_DATA_EACH_CLASS * 2
        for sent in self.test_pos_examples:
            predict = self.classify_sent(sent)
            if predict == 1:
                true_positives += 1
        for sent in self.test_neg_examples:
            predict = self.classify_sent(sent)
            if predict == 0:
                true_negatives += 1
        false_positives = project_config.NUMBERS_TEST_DATA_EACH_CLASS - true_positives
        false_negative = project_config.NUMBERS_TEST_DATA_EACH_CLASS - true_negatives
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negative)
        F1_Score = (2 * precision * recall) / (precision + recall)
        print("True positive:", true_positives)
        print("False positive:", false_positives)
        print("True negative:", true_negatives)
        print("False negative:", false_negative)
        print("F1 Score:", F1_Score * 100)

    def execute(self):
        return self.container
