import torch
from flask import Flask

from engine.ai.models import torch_model, tf_model


OD_PATH = 'weights/RSNA_OD.h5'
MODEL_NAMES = ['resnet50', 'densenet201', 'mobilenet_v2']
OUTPUT_PATH = '../vis_files/output.png'
DEVICE = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
RESPONSE = {'message': "Prediction saved successfully!"}

def init (application: Flask) -> Flask:
    application.config ['OUTPUT_PATH'] = OUTPUT_PATH
    application.config ['OD_MODEL'] = tf_model (weight_path = OD_PATH)
    CLS_MODELS = {}
    for model_name in MODEL_NAMES:
        CLS_MODELS [model_name] = torch_model (
            model_name,
            DEVICE,
            'weights/' + model_name + '.pth')
    application.config ['CLS_MODELS'] = CLS_MODELS
    application.config ['DEVICE'] = DEVICE
    application.config ['RESPONSE'] = RESPONSE

    return application