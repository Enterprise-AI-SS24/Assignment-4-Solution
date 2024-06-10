from zenml import step
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing_extensions import Annotated

@step
def create_preprocessing_pipeline(dataset:pd.DataFrame,target:str) -> Pipeline:
    """
    Constructs a preprocessing pipeline for a dataset with specified target column.
    """
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    numerical_columns = dataset.select_dtypes(exclude=['object']).columns
    categorical_columns = categorical_columns.drop(target)
    
    numerical_pipeline = Pipeline([
        ("num_imputer",SimpleImputer(strategy="median")),
        ("scaler",MinMaxScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("cat_imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
    ])
    preprocessing = ColumnTransformer([
        ("numerical",numerical_pipeline,numerical_columns),
        ("categorical",categorical_pipeline,categorical_columns)
    ])
    
    pipeline = Pipeline([
        ("preprocessing",preprocessing)
    ])
    
    return pipeline
    
    