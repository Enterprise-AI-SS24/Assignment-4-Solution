from steps import hp_tuning,model_trainer,evaluate_model
from zenml import pipeline
from zenml.client import Client
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

@pipeline(enable_cache=False)
def training_pipeline():
    """ 
        Pipeline to train and deploy a machine learning model using preprocessed and encoded datasets.
    """
    client = Client()
    X_train = client.get_artifact_version("X_train_preprocessed")
    X_test = client.get_artifact_version("X_test_preprocessed")
    y_train = client.get_artifact_version("y_train_encoded")
    y_test = client.get_artifact_version("y_test_encoded")

    best_parameters = hp_tuning(X_train,y_train)
    model,in_sample_score =model_trainer(X_train,y_train,best_parameters)
    deploy = evaluate_model(model,X_test,y_test)
    mlflow_model_deployer_step(model=model,deploy_decision=deploy,workers=1)