import os
import cv2
import random
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, Rescaling, Resizing, RandomContrast

from detection.data_prepare import DataPrepare
from detection.model import Model

data_dir = "C:/Users/s2007/Documents/py/ML/covid/COVID-19_Radiography_Dataset/"
catagories = ["COVID", "Normal", "Viral Pneumonia"]
img_width = 384
img_height = 384
img_depth = 1
batch_size = 32
augment = False
dropout = True
epochs = 10

# process data

# data = DataPrepare(data_dir, catagories, img_width, img_height)
# data.showExample(resize=True)
# data.make_train_validate("x_real.pickle", "y_real.pickle", grey_scale=True)

# start training

# print("Build Model")
# covid_model = Model(data_dir, catagories, img_width, img_height, img_depth, batch_size, augment, dropout) # may need change decay steps
# print("Start Training")
# covid_model.train(x="x_real.pickle", y="y_real.pickle", epochs=epochs) # %tensorboard --logdir logs/fit

# retrieve model

covid_model = tf.keras.models.load_model('20210904-122829_model') # good: 20210829-205253_model normal good: 20210904-122829_model
covid_model.summary()
x = pickle.load(open("x_test.pickle", "rb"))
y = pickle.load(open("y_test.pickle", "rb"))
x = np.array(x).reshape(-1, img_width, img_height, img_depth)
y = np.array(y)

# predict

predictions = covid_model.predict(x, x.shape[0])
y_pred_list = np.argmax(predictions, axis=-1)
accuracy = (y_pred_list == y).mean()
print(f"accuracy: {accuracy}")
