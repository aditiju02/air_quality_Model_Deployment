# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:51:25 2023

@author: aditi
"""

import numpy as np
from flask import Flask , request , jsonify, render_template
from gevent.pywsgi import WSGIServer
import pickle

# create flask app
aqiapp = Flask(__name__, template_folder="Templates", static_folder="static")


# load the pickle model
model = pickle.load(open("model.pkl","rb"))

@aqiapp.route("/")
def Home():
    return render_template("index.html")

@aqiapp.route("/predict", methods=["POST"] )
def predict():
    float_features = [float(x) for x in request.form.values() ]
    features = [np.array(float_features)]
    prediction = model.predict(features) 
    return render_template("index.html", prediction_text = "The AQI is {}".format(prediction))


if __name__ == "__main__":
    aqiapp.run(debug=True, port=8080)
    #from Waitress import serve
    #serve(apiapp, host="0.0.0.0", port=8080)
    #aqiapp.run(debug=True)
    #http_server = WSGIServer(('', 5000), aqiapp)
    #http_server.serve_forever()
