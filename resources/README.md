# Resources

## Dataset

RSNA is a Chest X-Ray image dataset contain over 25k images and over than 30k labels and bounding boxes for Lung Opacity detection

## Folder info

#### Classification_Torch

  * contain 5 files: first notebook file along with 3 python file for train RSNA dataset on 3 pretrained models on imagenet dataset including **ResNet50**, **DenseNet201** and **MobileNetV2**.
  * The notbook implemented for classification of Lung Opacity and Healthy pathology using Pytorch framework
  * other notebook file is testing trained model on other dataset


#### Classification_Lightning**

  * contain 3 files: a notebook file along with 2 python file. this part applied for just like previous part by adding **GoogleNet** model and **mlflow**


#### Object detection**

  * contain 5 files: a notebook and along with 4 python file for training RSNA dataset on custom model 
  * The notebook implemented for detection of Pneumonia using Tensorflow framework