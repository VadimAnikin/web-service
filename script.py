# importing libraries
import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask, render_template, request
# creating instance of the class
app = Flask(__name__)


# to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# prediction function
def ValuePredictor(to_predict_list):
    clm = ['Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category',
       'Leather interior', 'Fuel type', 'Engine volume', 'Mileage',
       'Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color',
       'Airbags']

    to_predict = pd.DataFrame(np.array(to_predict_list).reshape(1, 16), columns = clm)
    loaded_model = pickle.load(open("model_car.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        for i in range(len(to_predict_list)):
            if to_predict_list[i].isdigit():
                to_predict_list[i] = int(to_predict_list[i])
        result = ValuePredictor(to_predict_list)

        prediction = result

        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)