# ArfGAP1 Web App
This repo contains all the necessary script and instruction for the "Quantitative analysis of the cellular toxicity by ArfGAP1 protein using Deep Learning" project. Mainly this repo is for the demonstration purpose of the webapp which is the outcome of this project.
 
## Basic Instruction for running the web app

1. Open the app directory in your terminal
2. run python app.py
3. open http://0.0.0.0:8080/ in your chrome browser
4. Now you are on the landing page of the webapp. Here on the left side you will find Nucleus and Golgi image upload button. Just Upload a set of images.
5. Click on Predict button
Now wait for the result. It may take several minitue based on the machine configuration.

## Package Requirement
To run this webapp on your local machine you need to install following python pakages-  
caffe2==0.8.1  
cityscapesscripts==2.2.0  
cloudpickle==1.6.0  
docutils==0.18  
Flask==1.1.2  
fvcore==0.1.5.post20210423  
matplotlib==3.2.2  
mock==4.0.3  
numexpr==2.7.3  
numpy==1.20.1  
onnx==1.10.1  
opencv_python==4.5.1.48  
Pillow==8.4.0  
psutil==5.8.0  
pycocotools==2.0.2  
PyYAML==6.0  
recommonmark==0.7.1  
requests==2.22.0  
scipy==1.7.1  
setuptools==45.2.0  
Shapely==1.8.0  
Sphinx==4.2.0  
sphinx_rtd_theme==1.0.0  
tabulate==0.8.9  
termcolor==1.1.0  
torch==1.8.1+cpu  
torchvision==0.9.1+cpu  
tqdm==4.60.0  
Werkzeug==1.0.1  

Don't worry you will find a requirements.txt file to install all this via pip.
