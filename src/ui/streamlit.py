import cv2
import torch
from PIL import Image
import jsonpickle
import requests
import streamlit as st


# http://127.0.01:5000/ is from the flask api
server_url = "http://127.0.01:5000/"
predict_url = "http://127.0.01:5000/predict"
input_path = '../../vis_files/input.png'
output_path = '../../vis_files/output.png'

response = requests.post(server_url)
predict_response = requests.post(predict_url)

st.set_page_config(layout='wide')

col1, col2, col3 = st.columns([4, 10, 5])
    
col2.title ('Pneumonia Detection')
col2.write ('**Chest X-Ray**')


# select a mode
mode = col1.selectbox ('Select mode', [None, 'Detection', 'Classification'])

# insering help button
col3.subheader ('**Help**')

help_key = col3.button('Click here')
if help_key:
    col3.info ('''
        1. upload your Chest X-Ray image
        2. select your prediction mode and **Accept** it
        3. if your mode is classification you should choose your model too
        4. to disapear help notes click **Exit** button
        ''')
    Exit = col3.button('Exit')


def image_encoder (file, input_path, url):

    file = Image.open (file)
    file.save (input_path)

    headers = {'content_type': 'image/png'}

    with open (input_path, 'rb') as f:
        encoded_img = f.read ()

    return requests.post (
        url,
        data = encoded_img,
        headers = headers,
        timeout = 5000)



# select a pic
file = col2.file_uploader ("Please upload an image file", type = ['jpg', 'png'])

response_code = 100
response_text = None

if file is None:
    col2.text ('You didnt uploaded file yet!')


else:
    with col1.form (key = 'Prediction'):
            if mode == 'Classification':
                model_name = col1.radio ('Choose a model', ['resnet50', 'densenet201', 'mobilenet_v2'])

                response = requests.post(server_url + mode + '/' + model_name,
                                    timeout=5000)
                col1.write(response.content.decode("utf-8"))
                # inserting accept button
                accept = st.form_submit_button (label = 'Accept')
                if accept:
                    responsed = image_encoder (file, input_path, predict_url)
                    col1.image(input_path, use_column_width=True)
                    if responsed.status_code == 200:
                        response_text = 'output saved'


            elif mode == 'Detection':
                response = requests.post(server_url + mode, timeout=5000)
                col1.write(response.content.decode("utf-8"))
                # inserting accept button
                accept = st.form_submit_button (label = 'Accept')
                if accept:
                    responsed = image_encoder (file, input_path, predict_url)
                    col1.image(input_path, use_column_width=True)
                    if responsed.status_code == 200:
                        response_text = 'output saved'

            else:
                accept = st.form_submit_button(label = 'Accept')
                if accept:
                    pass


    if response_text == 'output saved':
        col2.image(output_path, use_column_width=True)