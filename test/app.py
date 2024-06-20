import utils_python as utils
import pandas as pd

def test_prediction():
    data = pd.read_csv('./data/diamonds_1.csv')
    data = data.sample(1)
    model_name = 'xgboost_diamonds'
    result = utils.predict_value(data, model_name)
    print(f"Predicted value: {result}")
    print(f"Actual value: {data['price'].values[0]}")

if __name__ == "__main__":
    test_prediction()