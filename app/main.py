
import csv
import numpy as np
import math
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from .model import load_model, predict_image, predict_price
from .preprocess import prepocess_image

from .includes import *

# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

DATA = {}

csv.field_size_limit(sys.maxsize)

def load_data(file_data_full: str, data: dict):
    if not data:
        d_data = {}
        with open(file_data_full, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, row in enumerate(data):
                if i == 0:  # entêtes de colonnes CSV
                    continue
                model = DEFAULT_MODEL_KEY
                d_data[model] = {}
                for j, e in enumerate(row):
                    if j < IDX_TITLE:
                        d_data[model][MAP_DATA_FULL[j]] = e
                    elif j == IDX_TITLE:
                        model = e
                        d_data[model] = {}
                        d_data[model].update(d_data[DEFAULT_MODEL_KEY])
                        d_data[DEFAULT_MODEL_KEY] = {}
                    else:
                        d_data[model][MAP_DATA_FULL[j]] = e
        DATA = d_data
    else:
        DATA = data
    return DATA

class Prediction(BaseModel):
    Model: str
    Classe: int
    Proba: dict
    Price: int
    Data: dict

app = FastAPI()

model = load_model(MODEL_DIR + '/cars_finetuning.keras')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification API"}

@app.post("/predict_image",response_model=Prediction)
async def predict_image_endpoint(year: int, km: int, energy: str, transmission: str, power: int, file: UploadFile = File(...)):
    try :
        image = await file.read()
        preprocessed_image = prepocess_image(image)
        prediction = predict_image(model, preprocessed_image)
        round_proba = {i : round(p, 5) for i, p in enumerate(prediction[0])}

        number = np.argmax(prediction[0])
        car_model = CARS[int(number)].replace('_', ' ')
        data = load_data(FILE_SPECS, DATA)[car_model]
        price = predict_price(car_model, year, km, energy, transmission, power)

        return {
            "Model" : car_model,
            "Classe": number,
            "Proba" : round_proba,
            "Price" : math.ceil(price),
            "Data"  : data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


