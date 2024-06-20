import datetime
import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

def save_model(model, model_name,preferred_preprocessing=0):
    # preferred_preprocessing: 
    # 0 for standard preprocessing preferring boolean and numerical values
    # 1 for preprocessing preferring boolean and categorical values, trees and boosting models
    
    #create variable containg the timestamp in form of string, including date_time
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M")

    model_dict = { 'model': model, 'model_name': model_name , 'preferred_preprocessing': preferred_preprocessing}
    model_path = '../history/current_models/'
    old_model_path = '../history/old_models/'
    # if there is a file with the same name, move it to the old folder without overwriting the old models 
    try:
        with open(model_path + model_name + '.pkl', 'rb') as f:
            old_model = pickle.load(f)
        with open(old_model_path + model_name + timestamp + '.pkl', 'wb') as f:
            pickle.dump(old_model, f)
    except:
        pass

    with open(model_path + model_name + '.pkl', 'wb') as f:
        pickle.dump(model_dict, f)

def save_scores(y_test, y_pred, model_name):
    scores = { 'Model name:' : model_name,'r2_score': r2_score(y_test, y_pred), 'mae': mean_absolute_error(y_test, y_pred) }
    scores_path = '../history/current_scores/'
    old_scores_path = '../history/old_scores/'
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M")

    print(model_name)
    print(scores)
    # if there is a file with the same name, move it to the old folder without overwriting the old scores
    try:
        with open(scores_path + model_name + '.csv', 'r') as f:
            old_scores = pd.read_csv(f)
        old_scores.to_csv(old_scores_path + model_name + timestamp +'.csv', index=False)
    except:
        pass

    # save the scores to disk as csv
    pd.DataFrame(scores, index=[0]).to_csv(scores_path + model_name + '.csv', index=False)