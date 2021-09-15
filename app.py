from flask import Flask, render_template, request, jsonify, url_for
import requests
import pickle
import numpy as np
import sklearn


def load_model():
    with open('./model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html',)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    PM2_5 = int(float(request.form['PM2.5']))
    NO2 = int(float(request.form['NO2']))
    CO = int(float(request.form['CO']))
    SO2 = int(float(request.form['SO2']))
    O3 = int(float(request.form['O3']))
    data = np.array([[PM2_5, NO2, CO, SO2, O3]])
    output = model.predict(data)
    return render_template("result.html", prediction=output)


if __name__ == "__main__":
    app.run(debug=True)
