import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: PatientData):
    prediction = model.predict([[data.Pregnancies, data.Glucose, data.BloodPressure,
                                  data.SkinThickness, data.Insulin, data.BMI,
                                  data.DiabetesPedigreeFunction, data.Age]])
    return {"prediction": int(prediction[0])}
