from zenml import pipeline
from zenml.client import Client
from steps import prediction_service_loader, predictor, inference_data_loader, inference_preprocessing

@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Runs the inference pipeline to predict the target variable of the inference data
    """
    data = inference_data_loader("./data/inference.csv")
    client = Client()
    preprocessing_pipeline = client.get_artifact_version("pipeline")
    preprocessed_data = inference_preprocessing(preprocessing_pipeline,data)    
    model_deployment_service = prediction_service_loader(
        pipeline_name="training_pipeline",
        step_name="mlflow_model_deployer_step",
    )
    prediction = predictor(service=model_deployment_service, input_data=preprocessed_data)
    print("adwawdawd",prediction)