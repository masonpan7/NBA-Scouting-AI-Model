from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

class PredictionInput(BaseModel):
    name: str
    year: float
    age: float
    is_first_year: float
    pos: str
    g: float
    mp: float
    mpg: float
    usg: float
    ts: float
    fga: float
    fg: float
    trb: float
    trb_g: float
    ast: float
    ast_g: float
    pts: float
    pts_g: float
    ws: float
    ws_48: float
    bpm: float
    vorp: float
    threepar: float
    stl: float
    stl_g: float
    tov: float
    tov_g: float
    mpg_change: float
    ast_change: float
    trb_change: float
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the NBA Scouting AI API. Use /docs for API documentation."}

@app.post('/predict')
def predict(input: PredictionInput):
    try:
        # Convert input to DataFrame
        features = pd.DataFrame([input.dict().values()], columns=input.dict().keys())
        features = features.drop(columns=['name'])

        # Rename columns to match the model's feature names
        features.rename(columns={
            'year': 'Year',
            'age': 'Age',
            'is_first_year': 'is_first_year',
            'pos': 'Pos',
            'g': 'G',
            'mp': 'MP',
            'mpg': 'MP/G',
            'usg': 'USG%',
            'ts': 'TS%',
            'fga': 'FGA',
            'fg': 'FG%',
            'trb': 'TRB',
            'trb_g': 'TRB/G',
            'ast': 'AST',
            'ast_g': 'AST/G',
            'pts': 'PTS',
            'pts_g': 'PTS/G',
            'ws': 'WS',
            'ws_48': 'WS/48',
            'bpm': 'BPM',
            'vorp': 'VORP',
            'threepar': '3PAr',
            'stl': 'STL',
            'stl_g': 'STL/G',
            'tov': 'TOV',
            'tov_g': 'TOV/G',
            'blk': 'BLK',
            'blk_g': 'BLK/G',
            'threepa_g': '3PAr',
            'mpg_change': 'MP/G_change',
            'ast_change': 'AST_change',
            'trb_change': 'TRB_change',
            'pts_g_change': 'PTS/G_change',
            'ws_change': 'WS_change',
            'ws_48_change': 'WS/48_change',
            'per_change': 'PER_change',
            'stl_change': 'STL_change',
            'usg_change': 'USG%_change',
            'ts_change': 'TS%_change',
            'fg_change': 'FG%_change',
            'tovg_change': 'TOV/G_change',
            'threepa_g_change': '3PA/G_change',
            'blk_change': 'BLK_change'
        }, inplace=True)

        def define_breakout(df):
            df['breakout'] = 0

            for i in range(len(df)):
                pos = df.loc[i, 'Pos']
                ppg_change = df.loc[i, 'PTS/G_change']
                per_change = df.loc[i, 'PER_change']
                ws_change = df.loc[i, 'WS_change']
                trb_change = df.loc[i, 'TRB_change']
                blk_change = df.loc[i, 'BLK_change']
                ast_change = df.loc[i, 'AST_change']
                stl_change = df.loc[i, 'STL_change']
                ts_change = df.loc[i, 'TS%_change']
                usg_change = df.loc[i, 'USG%_change']
                mp_change = df.loc[i, 'MP/G_change']
                tov_change = df.loc[i, 'TOV/G_change']
                fg_change = df.loc[i, 'FG%_change']
                threepa_change = df.loc[i, '3PA/G_change']
                ws48_change = df.loc[i, 'WS/48_change']

                criteria_met = 0

                if pos == 'PG':
                    if ppg_change >= 4.0:
                        criteria_met += 1
                    if ast_change >= 1.8:
                        criteria_met += 1
                    if ts_change >= 2.0:
                        criteria_met += 1
                    if ws_change >= 2.0:
                        criteria_met += 1
                    if usg_change >= 3.0:
                        criteria_met += 1
                    if tov_change <= 0.8:
                        criteria_met += 1
                    if mp_change >= 5.5:
                        criteria_met += 1
                    if stl_change >= 0.5:
                        criteria_met += 1

                elif pos == 'SG':
                    if ppg_change >= 3.5:
                        criteria_met += 1
                    if fg_change >= 1.8:
                        criteria_met += 1
                    if threepa_change >= 1.5:
                        criteria_met += 1
                    if ts_change >= 2.5:
                        criteria_met += 1
                    if ast_change >= 1.3:
                        criteria_met += 1
                    if ws_change >= 1.8:
                        criteria_met += 1
                    if mp_change >= 5.0:
                        criteria_met += 1
                    if tov_change <= 0.7:
                        criteria_met += 1

                elif pos == 'SF':
                    if ppg_change >= 3.5:
                        criteria_met += 1
                    if trb_change >= 1.8:
                        criteria_met += 1
                    if ast_change >= 1.5:
                        criteria_met += 1
                    if fg_change >= 1.5:
                        criteria_met += 1
                    if ts_change >= 2.0:
                        criteria_met += 1
                    if stl_change >= 0.4:
                        criteria_met += 1
                    if ws_change >= 2.0:
                        criteria_met += 1
                    if mp_change >= 5.0:
                        criteria_met += 1

                elif pos == 'PF':
                    if ppg_change >= 4.5:
                        criteria_met += 1
                    if trb_change >= 2.0:
                        criteria_met += 1
                    if blk_change >= 0.6:
                        criteria_met += 1
                    if fg_change >= 1.8:
                        criteria_met += 1
                    if ts_change >= 2.5:
                        criteria_met += 1
                    if ws_change >= 2.0:
                        criteria_met += 1
                    if mp_change >= 5.5:
                        criteria_met += 1
                    if tov_change <= 0.7:
                        criteria_met += 1

                elif pos == 'C':
                    if trb_change >= 2.5:
                        criteria_met += 1
                    if blk_change >= 0.7:
                        criteria_met += 1
                    if fg_change >= 2.0:
                        criteria_met += 1
                    if ppg_change >= 3.5:
                        criteria_met += 1
                    if ts_change >= 2.5:
                        criteria_met += 1
                    if ws48_change >= 0.03:
                        criteria_met += 1
                    if mp_change >= 5.0:
                        criteria_met += 1
                    if tov_change <= 0.4:
                        criteria_met += 1


                if criteria_met >= 4:
                    df.loc[i, 'breakout'] = 1
                

            return df

        df_features = define_breakout(features)
        df_features = df_features.drop(columns=['Pos'])

        # Define the model features
        model_features = model.get_booster().feature_names
        df_features = df_features[model_features]

        # Standardize the data
        df_features = pd.DataFrame(scaler.transform(df_features), columns=model_features)

        # Weight important features
        weighted_features = ['MP/G_change', 'USG%_change', 'PTS/G_change', 'WS/48_change', 'G', 'BPM']
        for feature in weighted_features:
            df_features[feature] *= 2


        # Make prediction
        probability = model.predict_proba(df_features)[:, 1]
        probability = round((float(probability[0]) * 100), 2)

        # Return result
        # Add confidence level
        if probability >= 75:
            confidence = "high"
        elif probability >= 50:
            confidence = "moderate"
        else:
            confidence = "low"


        # Return enhanced result
        # Return enhanced result as plain text
        return (
            f"{input.name} ({input.pos}) has a {probability}% chance of having a breakout season next year."
        )
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "An internal error occurred. Please check the server logs."}
    