import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def inference_data_loader(filename: str) -> Annotated[pd.DataFrame,"input_data"]:
    """ Loads a CSV File and transforms it to a Pandas DataFrame
    """
    data = pd.read_csv(filename,index_col="Date")
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = data.loc[[today]]
    return data