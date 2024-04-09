from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager

#we create a class that it will be provided to FAastAPi to inpret the input data 
class FeatureSet(BaseModel):
    carat: float
    cut: float
    color: float
    clarity: float
    depth: float
    table: float
    x: float
    y: float
    z: float

#we define a function that receive a feature set x and return a model response 
def regressor(x:dict)->dict:
    with open('../diamond_linear_tree_regression.pkl','rb') as model_file:
        loaded_model=pickle.load(model_file)
    x_df=pd.DataFrame(x,index=[0])
    res=loaded_model.predict(x_df)[0]
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
    return ml_models["regressor"](feature_set.model_dump())

