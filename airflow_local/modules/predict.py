import json
import os

import dill
import pandas as pd

model_filename = '../data/models/cars_pipe_202212280711.pkl'
path_to_json = '../data/test/'

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

def predict():
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    df_prediction = pd.DataFrame(columns=['car_id', 'pred'])

    for json_file in json_files:
        with open(path_to_json + json_file, encoding='utf-8') as f:
            form = json.load(f)
            df = pd.DataFrame([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_prediction = pd.concat([df_prediction, df1], axis=0)

    df_prediction.to_csv(f'../data/predictions/prediction.csv')


if __name__ == '__main__':
    predict()
