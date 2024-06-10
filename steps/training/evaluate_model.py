import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import mlflow

@step(experiment_tracker="mlflow_experiment_tracker")
def evaluate_model(model:ClassifierMixin,X_test:pd.DataFrame,y_test:pd.DataFrame) -> Annotated[bool,"deployment_decision"]:
    """
    Evaluates the trained model and returns a deployment decision based on the out-of-sample accuracy.
    """
    prediction = model.predict(X_test)
    score = model.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    precision = precision_score(y_test,prediction)
    f1 = f1_score(y_test,prediction)
    mlflow.log_metric("Recall",recall)
    mlflow.log_metric("Precision",precision)
    mlflow.log_metric("F1_Score",f1)
    mlflow.log_metric("Out_of_Sample_Accuracy",score)
    if score > 0.8:
        deploy = True
    else:
        deploy = False
    return deploy