import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: pd.DataFrame,
) -> np.ndarray:

    """Run an inference request against a prediction service"""
    service.start(timeout=10) 
    prediction = service.predict(input_data)
    return prediction