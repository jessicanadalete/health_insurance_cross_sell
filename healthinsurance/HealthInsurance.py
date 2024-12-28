import pickle
import inflection
import numpy as np
import pandas as pd


class HealthInsurance( object ):
  def __init__( self ):
    self.home_path = '/opt/render/project/src/parameter/'
    self.annual_premium_standard              = pickle.load( open( self.home_path + 'annual_premium.pkl', 'rb') )
    self.age_scaler                           = pickle.load( open( self.home_path + 'age.pkl', 'rb') )
    self.vintage_scaler                       = pickle.load( open( self.home_path + 'vintage.pkl', 'rb') )

  def clean_data (self, df1):

    old_columns = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code',
                'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
                'Policy_Sales_Channel', 'Vintage', 'Response']

    snakecase = lambda x: inflection.underscore(x)
    new_columns = list(map(self.snakecase, old_columns))

    df1.columns = new_columns

    return df1

  def data_preparation (self, df5):

    #standardization
    df5['annual_premium'] = self.stand_annual_premium.fit_transform(df5[['annual_premium']].values)

    #recasling
    df5['age'] = self.age_scaler.fit_transform(df5[['age']].values)
    
    df5['vintage'] = self.vintage_scaler.fit_transform(df5[['vintage']].values)

    #vehicle_damage (Label encoding)
    df5['vehicle_damage'] = df5['vehicle_damage'].apply(lambda x: 1 if x == "Yes" else 0)

    #gender (Label encoding)
    mean_encoded_gender = df5.groupby('gender')['response'].mean()
    df5['gender'] = df5['gender'].map(self.mean_encoded_gender)

    #vehicle age transformation (Order encoding)
    df5['vehicle_age'] = df5['vehicle_age'].apply(lambda x: 0 if x == "< 1 Year" else 1 if x== "1-2 Year" else 2)

    #region_code (Targeting encoding - multivariables)
    mean_encoded_region = df5.groupby('region_code')['response'].mean()
    df5['region_code'] = df5['region_code'].map(self.mean_encoded_region)

    #previously insurance (Label encoding)
    df5['previously_insured'] = df5['previously_insured'].apply(lambda x: 1 if x == "Yes" else 0)

    #policy sales channel (Targeting encoding - multivariables)
    mean_encoded_channel = df5.groupby('policy_sales_channel').size()/len(df5)
    df5['policy_sales_channel'] = df5['policy_sales_channel'].map(self.mean_encoded_channel)

    #feature selection
    cols_selected = ['vintage','annual_premium','age', 'vehicle_damage', 'region_code', 'policy_sales_channel', 'vehicle_age']

    return df5[cols_selected]

  def get_prediction( self, model, original_data, test_data ):
    #prediction
    pred = model.predict_proba( test_data )

    #join pred into the original data
    original_data['prediction'] = pred[:, 1].tolist()

    return original_data.to_json( orient='records', date_format='iso' )
