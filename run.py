from bean.Container import Container
from pipeline.load_data import LoadData
from pipeline.preprocess_test_data import PreprocessTestData
from pipeline.preprocess_train_data import PreprocessTrainData
from pipeline.preprocess_predict_data import PreprocessPredictData
from pipeline.model import Model


def main():
    container = Container()
    container = LoadData(container).execute()
    container = PreprocessTrainData(container).execute()
    container = PreprocessTestData(container).execute()
    container = PreprocessPredictData(container).execute()
    my_model = Model(container)
    my_model.evaluate_model()
    # my_model.predict()


if __name__ == '__main__':
    main()
