from grpc import method_handlers_generic_handler
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# create a flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text = 'The flower species is {}'.format(prediction))

if  __name__ == '__main__':
    app.run(debug = True)
