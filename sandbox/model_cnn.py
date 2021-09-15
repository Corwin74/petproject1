"""
Predictcs a number
"""
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt


def normalize(img):
    image = img.copy()
    image = image.astype('float32')
    image = image.reshape(28, 28, 1)
    image /= 255
    return image

def predict(img):
    model = load_model('cnn.hdf5')
    pred = model.predict(img)
    return pred

