from zenml import step
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import optuna
import pandas as pd
from functools import partial
from typing_extensions import Annotated

def objective(trial,X_train,y_train):
       """ 
              This function defines the objective for the Optuna optimization.
       """
       # Define the hyperparameters to tune
       max_depth = trial.suggest_int('max_depth', 1, 30)
       min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
       min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
       criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
       # Define the model
       model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
       )
       # Split the data
       X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,shuffle=False)
       # Train the model
       model.fit(X_train, y_train)
       # Evaluate the model
       accuracy = accuracy_score(y_test, model.predict(X_test))
       return accuracy

@step
def hp_tuning(X_train: pd.DataFrame, y_train: pd.Series,trials:int=100)-> Annotated[dict,"Best hyperparameters"]:
   """
   This step tunes the hyperparameters of a Decision Tree model using Optuna.
   """
   obj = partial(objective,X_train=X_train,y_train=y_train)
   # Create a study
   study = optuna.create_study(direction="maximize")

   # Optimize the study
   study.optimize(obj, n_trials=trials)
   # Get the best hyperparameters
   best_params = study.best_params
   print(best_params,type(best_params))
   # Return the best hyperparameters
   return best_params