
import pickle
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, responses
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: str
    max_torq: float
    torque_npm: float

def regressor(x: dict) -> float:
    with open("model.pickle", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.DataFrame(x, index=[0])
    res = loaded_model.predict(x_df)[0]
    return round(res, 1)

def file_regressor(x):
    with open("model.pickle", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.read_csv(x, index_col=[0])
    x_df['seats'] = x_df['seats'].astype(object)
    x_df['predict'] = loaded_model.predict(x_df)
    x_df.to_csv('result.csv')

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["regressor"] = regressor
    ml_models["file_regressor"] = file_regressor
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/result/")
async def download_result():
    return responses.FileResponse(path='result.csv', filename='Результат.csv')

@app.post("/predict_item")
def predict_item(item: Item):
    return ml_models["regressor"](item.model_dump())

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        #with open(file.filename, 'wb') as f:
        with open(f'download_{file.filename}', 'wb') as f:
            f.write(contents)
        ml_models["file_regressor"](f'download_{file.filename}')
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    return f"{file.filename} успешно загружен и обработан, загрузите результат по ссылке http://127.0.0.1:8000/result"
    
