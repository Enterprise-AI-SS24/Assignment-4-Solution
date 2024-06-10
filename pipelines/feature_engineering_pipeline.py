from steps import loading_data,split_data,label_encoding,create_preprocessing_pipeline,feature_preprocessor
from zenml import pipeline

@pipeline
def feature_engineering_pipeline():
    """"
        Pipeline function for performing feature engineering on weather data.
    """
    dataset = loading_data("./data/Weather_Perth.csv")
    X_train,X_test,y_train,y_test = split_data(dataset,"RainTomorrow")
    pipeline = create_preprocessing_pipeline(dataset,"RainTomorrow")
    X_train,X_test,pipeline = feature_preprocessor(pipeline,X_train,X_test)
    y_train,y_test = label_encoding(y_train,y_test)
    