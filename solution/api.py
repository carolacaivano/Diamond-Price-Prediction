from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sklearn.preprocessing import StandardScaler

#we create a class that it will be provided to FAastAPi to inpret the input data 
class FeatureSet(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

def preprocess_features(features: FeatureSet) -> pd.DataFrame:
    """Function to preprocess input features"""
    # Convert the input features into a DataFrame
    features_df = pd.DataFrame([features.model_dump()])
    cut_order= {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    clarity_order={'I1':0,'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7}
    color_order={'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5,'D':6}
    # Label encoding
    features_df['encoded_cut'] = features_df['cut'].map(cut_order)
    features_df['encoded_clarity'] = features_df['clarity'].map(clarity_order)
    features_df['encoded_color'] = features_df['color'].map(color_order)
    #We drop the columns cut, clarity and color
    features_df=features_df.drop(['cut'],axis=1)
    features_df=features_df.drop(['clarity'],axis=1)
    features_df=features_df.drop(['color'],axis=1)
    features_df= StandardScaler().fit_transform(features_df)
    
    return features_df    

#we define a function that receive a feature set x and return a model response 
def regressor(x:dict)->dict:
    with open('../diamond_linear_tree_regression.pkl','rb') as model_file:
        loaded_model=pickle.load(model_file)
    res=loaded_model.predict(x)[0]
    return{"prediction": res}    

#we ulpoad the model
ml_models={}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["regressor"]=regressor
    yield
    ml_models.clear()


app=FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict")
async def predict(feature_set: FeatureSet):
    #preprocess the input features
    preprocessed_features = preprocess_features(feature_set)
    return ml_models["regressor"](preprocessed_features)

