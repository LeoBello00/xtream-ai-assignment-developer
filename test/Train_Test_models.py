# import every model using pickle into a dictionary, by importing every pickle in the models folder
import pandas as pd
import pickle,os
from sklearn.model_selection import train_test_split
import numpy as np
import utils

    

from copy import deepcopy
def preprocess_input_for_tree_models(input_data):
    diamonds_processed = deepcopy(input_data)
    # remove every line with a missing value
    diamonds_processed = diamonds_processed[(diamonds_processed.x * diamonds_processed.y * diamonds_processed.z != 0) & (diamonds_processed.price > 0)]
    diamonds_processed['cut'] = pd.Categorical(diamonds_processed['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
    diamonds_processed['color'] = pd.Categorical(diamonds_processed['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    diamonds_processed['clarity'] = pd.Categorical(diamonds_processed['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    return diamonds_processed

def preprocess_input_for_linear_models(input_data):
    diamonds_processed = deepcopy(input_data)
    diamonds_processed = diamonds_processed[(diamonds_processed.x * diamonds_processed.y * diamonds_processed.z != 0) & (diamonds_processed.price > 0)]
    diamonds_processed = diamonds_processed.dropna()
    diamonds_processed.drop(columns=['depth', 'table', 'y', 'z'])
    diamonds_processed = pd.get_dummies(diamonds_processed, columns=['cut', 'color', 'clarity'])
    return diamonds_processed
        


def fit_test_models(new_data):
    # Load the models from the models folder
    model_path = "./history/current_models"
    models = {}

    for file in os.listdir(model_path):
        if file.endswith(".pkl"):
            model_name = file.split(".")[0]
            model = pickle.load(open(f"{model_path}/{file}", "rb"))
            models[model_name] = model

    for model_name, model in models.items():
        print(f"Model: {model_name}")
        print(f"Model: {model}")
        print("Fitting models")
    x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(new_data[0].drop(columns=['price']), new_data[0]['price'], test_size=0.2, random_state=0)
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(new_data[1].drop(columns=['price']), new_data[1]['price'], test_size=0.2, random_state=0)

    for model_name,model in models.items():
        if(model["preferred_preprocessing"] == 0):
            model["model"].fit(x_train_0, np.log(y_train_0))
            utils.save_model(model["model"], model_name)
        elif(model["preferred_preprocessing"] == 1):
            model["model"].fit(x_train_1, y_train_1)
            utils.save_model(model["model"], model_name,1)
        else:
            print(model["preferred_preprocessing"])
            raise ValueError("Invalid preprocessing method")
    test_models((y_test_0, y_test_1, x_test_0, x_test_1),models)

def test_models(data,models):
    y_val_0, y_val_1,x_val_0,x_val_1 = data
    results = {}
    for model_name,model in models.items():
        if(model["preferred_preprocessing"] == 0):
            pred = np.exp(model["model"].predict(x_val_0))
            utils.save_scores(y_val_0, pred, model_name)
        elif(model["preferred_preprocessing"] == 1):
            pred = model["model"].predict(x_val_1)
            utils.save_scores(y_val_1, pred, model_name)
        else:
            print(model["preferred_preprocessing"])
            raise ValueError("Invalid preprocessing method")
    return results


