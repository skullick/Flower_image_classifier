# Flower Image Classifier
This project builds a deep learning model using transfer learning to classify images of flowers into different categories. The dataset contains images of various flower species, which is divided into training, validation, and testing sets to train and evaluate the model. This project uses the VGG-16 architecture pre-trained on ImageNet and fine-tunes it to recognize specific flower species. The code for the project was developed using Jupyter Notebook and converted to a command-line application.
## Installation
For this project, I have Python, PyTorch, torchvision, NumPy, matplotlib and other dependencies installed. You can install the required packages using ```pip```.
## Data
The dataset is organized into three directories: train, valid, test.
I have uploaded some sample images in each directory. You can find the full dataset [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
## Command Line Application
* To train a neural network using the dataset, use the ```train.py``` script. This script allows you to specify various training parameters, including the architecture, number of epochs, learning rate, and more.
* To use a trained model for classifying new images, use the ```predict.py``` script. This script loads a trained model checkpoint and makes predictions on a given image.
