import pandas as pd
import mlflow
from zenml import step
from typing_extensions import Annotated
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
from sklearn.base import ClassifierMixin
@step(experiment_tracker="mlflow_experiment_tracker")
def model_trainer(X_train: pd.DataFrame, y_train: pd.Series,best_parameters:dict)-> Tuple[Annotated[ClassifierMixin,"Model"],Annotated[float,"In_Sample_Accuracy"]]:
    """
    Trains a decision tree classifier model using the training dataset and the best hyperparameters found during hyperparameter tuning.
    """
    mlflow.sklearn.autolog()
    model = DecisionTreeClassifier(**best_parameters)
    model.fit(X_train,y_train)
    in_sample_score = model.score(X_train,y_train)
    return model,in_sample_score