import os
import re
import nltk
import string
import mlflow
import pickle
import dagshub
import numpy as np
import pandas as pd
from flask import Flask, render_template,request
from preprocessing_utility import normalize_text




# Set up the MLflow tracking URI
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO]: Loading environment variables from .env file")
    dagshub_token = os.getenv("DAGSHUB_PAT")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    print(f"[INFO] Latest model version: {latest_version[0].version}") if latest_version else print("[INFO] No model version found")
    return latest_version[0].version if latest_version else None

model_name = "model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    # show
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)