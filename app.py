from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

model = pickle.load(open('model.pkl', 'rb'))

class PredictionInput(BaseModel):
    name: str
    year: float
    age: float
    is_first_year: float
    g: float
    mp: float
    mpg: float
    usg: float
    ts: float
    fga: float
    fg: float
    trb: float
    trb_change: float
    ast: float
    ast_g: float
    pts: float
    pts_g: float
    ws: float
    ws_48: float
    bpm: float
    vorp: float
    threepar: float
    stl_g: float
    tov: float
    tov_g: float
    mpg_change: float
    ast_change: float
    pts_g_change: float
    ws_change: float
    ws_48_change: float
    per_change: float
    stl_change: float
    usg_change: float
    ts_change: float
    fg_change: float
    tovg_change: float
    threepa_g_change: float
    blk_change: float

@app.post('/predict')
def predict(input: PredictionInput):
    features = pd.DataFrame([input.dict().values()], columns=input.dict().keys())
    features = features.drop(columns=['name'])

    probabiltiy = model.predict_proba(features)[:, 1]

    return {"{name}'s ": input.name, 
        "probability of having a breakout season next year": probabiltiy[0]}