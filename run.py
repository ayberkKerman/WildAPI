from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import tensorflow as tf
from keras.models import load_model 
from tensorflow.keras.utils import load_img, img_to_array
import json

from numpy import argmax
from uuid import uuid4 

app = Flask(__name__)
model = load_model("F:\TezReact\API\Model.h5") #Write your document path here.
app.config['MAX_CONTENT_LENGTH'] = 1024*100
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.after_request
@cross_origin()
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,true')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response



@app.route('/predict', methods=["POST"])
@cross_origin()
def predict():
    data = request.files["img"] 
    img_name = str(uuid4()) 
    data.save(img_name)  
    data = img_to_array(load_img(img_name, color_mode="rgb", target_size=(224,224))).reshape(1,224, 224, 3)
    
    res = argmax(model.predict([data]))
    resp = res.tolist()
    return jsonify(resp) 


app.run()