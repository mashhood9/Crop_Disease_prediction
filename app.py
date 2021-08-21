from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet152V2.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf."
        advice="Please take necessary steps."
    elif preds==1:
        preds="The leaf is diseased cotton plant."
        advice="Please take necessary steps."
    elif preds==2:
        preds="The leaf is fresh cotton leaf."
        advice="No problems detected as of now, but keep checking every once in a while."
    else:
        preds="The leaf is fresh cotton plant."
        advice="No problems detected as of now, but keep checking every once in a while."

        
    
    
    return preds, advice


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file1']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(basepath,'static')):
            os.mkdir(os.path.join(os.getcwd(),'static'))
        file_path = os.path.join(
            basepath, 'static', secure_filename(f.filename))
        # file_path=os.path.join('D:\\Users',secure_filename(f.filename)) 
        f.save(file_path)

        print('file_path - ',file_path)

        # Make prediction
        preds, advice = model_predict(file_path, model)
        result={'res':preds, 'advice':advice, 'image_name':secure_filename(f.filename)}
        return render_template('results.html',result=result)
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)