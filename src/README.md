# Flask Api & Streamlit as UI

## Description

* An aplication which predict `Pneumonia` and Locates it.
* Frontend and Backend of app are `Streamlit` and `Flask` respectively
* User can choose each of predictions separately
* If Classification was chosen 3 models are as an Option contatin:
**resnet50**, **densenet201** and **mobilenet_v2**

## Usage

### Open the terminal

**clone the repository to local**

```bash
git clone https://github.com/miladlink/Streamlit_Flask
cd src
```

* for using app service should run first on  `http://localhost:5000`

```bash
chmod +x service.sh
./service.sh
```
* then streamlit runs for show app environment and usage on `http://localhost:8501`

```bash
chmod +x ui.sh
./ui.sh
```
* you want run both at the same time run below

```bash
chmod +x run.sh
./run.sh
```

## Results

**Upload An Image:**

![Screenshot from 2021-10-27 02-36-58](https://user-images.githubusercontent.com/81680367/139055613-63b0e661-96f7-4f53-ae58-f1b8749fe974.png)

**Get the Detection Result:**

![Screenshot from 2021-10-27 02-37-50](https://user-images.githubusercontent.com/81680367/139055623-1e14a6cb-c321-4894-99e3-c41dc88eba3d.png)
Get the Classification Result

![Screenshot from 2021-10-27 02-38-03](https://user-images.githubusercontent.com/81680367/139055640-fa00274a-c394-476c-aac9-ec0ac7909385.png)






