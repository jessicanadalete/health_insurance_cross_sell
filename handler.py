import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from flask_cors import CORS
from healthinsurance.HealthInsurance import HealthInsurance

# Loading model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'linear_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Instantiate HealthInsurance Class
pipeline = HealthInsurance()

app = Flask(__name__)

CORS(app) # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def healthinsurance_predict():
    
    test_json = request.get_json()
    
    if test_json:  # there is data
       
        if isinstance(test_json, dict):  # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())  # multiple examples

        try:
            # Data cleaning
            df1 = pipeline.clean_data(test_raw)

            # Feature engineering
            df2 = pipeline.data_preparation(df1)

            # Get prediction
            df_response = pipeline.get_prediction(model, test_raw, df2)

            # Return the response
            return Response(df_response, status=200, mimetype='application/json')
        
        except Exception as e:
            return Response(f"Error: {str(e)}", status=500, mimetype='application/json')

    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0')