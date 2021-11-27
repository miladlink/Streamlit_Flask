import jsonpickle
from flask import request, Response, jsonify

from engine.io.data import *
from engine.vis.visualize import *
from service import app


@app.route ('/')
def home_page ():
    return """ Backend of the App """

@app.route('/<mode>', methods = ['GET', 'POST'])
def mode(mode):
    global chosen_mode
    chosen_mode = mode
    return f"Choosen app is {mode}!\n\nHere's your input image:"

@app.route('/Classification/<name>', methods = ['GET', 'POST'])
def name (name):
    global chosen_mode
    chosen_mode = 'Classification'
    global model_name
    model_name = name

    return f"Choosen app is Classification using {name} model!\n\nHere's your input image:"

@app.route ('/predict', methods = ['POST'])
def prediction ():

    if chosen_mode == 'Classification':
        img = Bytes_to_torch (request.data)
        pred_cls = predict_and_gradcam (
            app.config ['CLS_MODELS'][model_name],
            img,
            app.config ['DEVICE'])
        pred_cls.savefig (app.config ['OUTPUT_PATH'])

    elif chosen_mode == 'Detection':
        img = Bytes_to_array (request.data)
        pred_det = show_preds_with_bboxes (
            app.config ['OD_MODEL'],
            img)
        pred_det.savefig (app.config ['OUTPUT_PATH'])

    response_pickled = jsonpickle.encode(app.config ['RESPONSE'])
    return Response(response = response_pickled, status = 200, mimetype = "application/json")