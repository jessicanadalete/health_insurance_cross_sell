import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from health_insurance.HealthInsurance import HealthInsurance

#loading model
with open('model/linear_model.pkl', 'rb') as file:
  model = pickle.load(file)

app = Flask( __name__ )

@app.route( '/health_insurance/predict', methods=['POST'] )

def health_insurance_predict():
    
    test_json = request.get_json()
    
    if test_json: # there is data
       
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0])
        else:
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys()) #multiple examples

        #instantiate HealthInsurance Class
        pipeline = HealthInsurance()

        #data cleaning
        df1 = pipeline.clean_data( test_raw )

        #feature enginnering
        df2 = pipeline.data_preparation( df1 )

        #prediction
        df_response = pipeline.get_prediction( model, test_raw, df2 )

        return df_response

    else:
      
      return Response( '{}', status=200, mimetype='application/json' )


if __name__ == '__main__':
    app.run( host='0.0.0.0', debug=True)