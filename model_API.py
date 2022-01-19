# model's imports
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI
from pydantic import BaseModel, confloat
import pickle

app = FastAPI()

#_____________________________________________________________________________________________
#Function for loading the model
@app.on_event("startup")
def load_model():
    global trained_model
    with open("Os_model.pkl", "rb") as my_stream:
        trained_model = pickle.load(my_stream)

#_____________________________________________________________________________________________
#Function for predicting and validating the model's inputs

class OfferStatus(BaseModel):
    time_h: int
    offer_type: int
    difficulty: int
    duration_h: int
    gender: int
    age: int
    income: float

@app.post("/predict")
def make_prediction(offer: OfferStatus):
    offer_dict = offer.dict()

    time_h = offer_dict["time_h"]
    offer_type = offer_dict["offer_type"]
    difficulty = offer_dict["difficulty"]
    duration_h = offer_dict["duration_h"]
    gender = offer_dict["gender"]
    age = offer_dict["age"]
    income = offer_dict["income"]
    

    predicted_status = trained_model.predict([[time_h, offer_type, difficulty, duration_h, gender, age, income]])
    if (predicted_status[0] == 0 ):
        return "Predicted Status: Offer Viewed"
    else: return "Predicted Status: Offer Completed"
