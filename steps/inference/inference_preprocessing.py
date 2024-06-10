from zenml import step
from sklearn.pipeline import Pipeline
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

@step
def inference_preprocessing(pipeline:Pipeline,dataset:pd.DataFrame,)-> Annotated[pd.DataFrame,"dataset_processed"]:
    """
    This function applies the preprocessing pipeline to the dataset and returns the transformed dataset.
    """
    dataset_transformed= pipeline.transform(dataset)
    cat_features_after_encoding = pipeline.named_steps['preprocessing'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(dataset.select_dtypes(include=['object']).columns)
    all_features = list(dataset.select_dtypes(exclude=['object']).columns )+ list(cat_features_after_encoding)
    dataset_df = pd.DataFrame(dataset_transformed, columns=all_features)
    return dataset_df