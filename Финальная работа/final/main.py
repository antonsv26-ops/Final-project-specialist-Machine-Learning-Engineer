import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('sber_pipe.pkl', 'rb') as file:
    data = dill.load(file)
pipeline = data['model']

class Sber(BaseModel):
    session_id: str
    visit_number: int
    hit_number: float
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_browser: str
    device_model: str
    device_screen_resolution: str
    geo_city: str
    geo_country: str


class Prediction(BaseModel):
    session_id: str
    Result: float


@app.post("/predict", response_model=Prediction)
def predict(form: Sber):
    raw_data = form.model_dump() if hasattr(form, 'model_dump') else form.dict()
    df_input = pd.DataFrame([raw_data])
    y = pipeline.predict(df_input)

    return{
        'session_id': form.session_id,
        'Result': y[0]
    }