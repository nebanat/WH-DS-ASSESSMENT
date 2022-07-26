# standard library imports 
import os
import time 
import pickle
import argparse

# third party library imports
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

# loads csv header format to check 
load_format_path = 'deployment/formats/load_format.csv' # csv header format
train_format_path = 'deployment/formats/train_format.csv' # csv header format for training
model_path_dict = {
    "lgb": "deployment/models/lgb.pkl",
    "xgb": "deployment/models/xgb.pkl"
}

prediction_folder = 'deployment/predictions/' # prediction are saved in this folder


def load_csv_headers(file_path:str) -> list:
    """
    summary:
        loads csv headers from file_path

    Args:
        file_path (str): path to csv file

    Raises:
        FileNotFoundError: if file_path does not exist
        NotImplementedError: if file_path is not a csv file

    Returns:
        list of columns (List): list of columns in csv file
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    
    if not file_path.endswith('.csv'):
        raise NotImplementedError(f'File must be a csv file: {file_path}')
    
    df = pd.read_csv(file_path)
    return list(df.columns)


def load_data(file_path:str) -> tuple:
    """
    summary: 
        loads data from file_path(csv file) and returns dataframe and y_true

    Args:
        file_path (str): path to csv file

    Raises:
        FileNotFoundError: file_path does not exist
        ValueError: file_path does not have the correct header format
        NotImplementedError: file is not a csv file

    Returns:
        tuple: (dataframe, y_true)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    
    if not file_path.endswith('.csv'):
        raise NotImplementedError(f'File must be a csv file: {file_path}')
    
    df = pd.read_csv(file_path)

    # loads csv header format for both training and load data
    load_headers = load_csv_headers(load_format_path)
    train_headers = load_csv_headers(train_format_path)

    # check if csv headers match
    if not df.columns.tolist() == load_headers:
        raise ValueError(f'Headers do not match: {df.columns.tolist()} != {load_headers}')
    
    # filter only training headers 
    df = df[train_headers]

    labels = df['y'].values
    df = df.drop(['y'], axis=1)


    return df, labels

def load_model(model_path:str) -> object:
    """
    summary: 
        loads model from model_path (pickle file)

    Args:
        file_path (str): path to model pickle file

    Raises:
        FileNotFoundError: if model_path does not exist

    Returns:
        object: loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict(file_path:str, model_type="xgb") -> None:
    """
    summary: 
        run prediction using model and data from file_path specified.
        Predictions are saved to prediction_folder.

    Args:
        file_path (str): path to csv file
        model_type (str): model type to use. default is xgb(xgboost)

    Raises:
        ValueError: if model_type is not in (lgb, xgb)

    Returns:
        None: saves predictions to prediction_folder
    
    """
    allowed_models = ("lgb", "xgb")
    if model_type not in allowed_models:
        raise ValueError(f"No model type {model_type}. allowable values are {allowed_models}")
    
    print("Loading data...")
    
    df, y_true = load_data(file_path)
    df = df.fillna(df.mean())

    print("Loading model...")
    model_path = model_path_dict.get(model_type)
    model = load_model(model_path)

    print("Predicting...")
    pred_start_time = time.time() # prediction start time
    
    if model_type =='xgb':
        y_pred = model.predict(xgb.DMatrix(df)) # convert to xgb.DMatrix before predictions
    else:
        y_pred = model.predict(df.values)

    pred_end_time = time.time() - pred_start_time # prediction end time

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("preparing predictions...")
    diff_pred = np.abs(y_true - y_pred) # prediction difference
    accuracy = np.count_nonzero(diff_pred <= 3) / len(diff_pred) # percent difference
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['diff_pred'] = diff_pred

    print("saving predictions...")
    ts = time.time() # hash timestamp to append to prediction file
    df.to_csv(prediction_folder + f'predictions_{model_type}_{ts}.csv', index=False)

    print(f'RMSE: {rmse:.2f}')
    print(f'Prediction time: {pred_end_time:.2f} seconds')
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    """ 
    command to run prediction:
        python deployment/predict.py --file-path {file_path} --model (lgb | xgb) 
    """
    parser = argparse.ArgumentParser(description="runs batch prediction")
    parser.add_argument('--file-path', type=str, required=True, help="path to the data file")
    parser.add_argument("--model", type=str, default='xgb', 
                            help="type of model either lgb or xgb --default=xgb")
    args = parser.parse_args()
    print(args.file_path)
    predict(args.file_path, model_type=args.model)
