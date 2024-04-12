import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel

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

def preprocess_features(features):
    """Function to preprocess input features"""
    if not isinstance(features, list):
        features = [features] 
    features_df = pd.DataFrame(features)

    # Define the encoding orders
    cut_order = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    clarity_order = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    color_order = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}

    # Label encoding
    features_df['encoded_cut'] = features_df['cut'].map(lambda x: cut_order.get(x, -1))
    features_df['encoded_clarity'] = features_df['clarity'].map(lambda x: clarity_order.get(x, -1))
    features_df['encoded_color'] = features_df['color'].map(lambda x: color_order.get(x, -1))

    # Drop the original categorical columns
    features_df = features_df.drop(['cut', 'clarity', 'color'], axis=1)

    return features_df

def medinc_regressor(x: dict) -> dict:
    """Function dedicated to prediction"""
    with open('./diamond_linear_tree_regression.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    # Preprocess the features
    x_df = preprocess_features(x)
    
    # Load the scaler
    with open('./scaler.pkl','rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)  

    # Apply scaler transformation
    x_scaled = loaded_scaler.transform(x_df)
    
    # Make the prediction
    res = loaded_model.predict(x_scaled)[0]
    
    return {"prediction": res}

# Creation of a context manager to manage the model's lifecycle
ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["medinc_regressor"] = medinc_regressor
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict")
async def predict(feature_set: FeatureSet):
    return ml_models["medinc_regressor"](feature_set.dict())

