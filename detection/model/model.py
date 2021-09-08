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

class Model():
    def __init__(self, data_dir, catagories, img_width: int, img_height: int, img_depth=1, batch_size=32, augment=False, dropout=False):
        self.data_dir = data_dir
        self.catagories = catagories
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.batch_size = batch_size
        self.augment = augment
        self.dropout = dropout

        self.num_classes = len(catagories)

        # Create and compile the model
        model=models.Sequential()

        if self.augment:
            data_augmentation = tf.keras.Sequential([
                RandomFlip('horizontal', input_shape=(self.img_height, self.img_width, self.img_depth)),
                RandomRotation(0.05),
                RandomContrast(0.1),
                RandomZoom((0,0.05), (0,0.05))
            ])
            model.add(data_augmentation)
            model.add(Rescaling(1./255))
        else:
            model.add(Rescaling(1./255, input_shape=(self.img_width, self.img_height, self.img_depth)))

        model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D((2,2)))

        if self.dropout:
            model.add(Dropout(0.5))

        model.add(Flatten())


        model.add(Dense(512, activation="relu"))

        if dropout:
            model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        #decaying learning rate
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=300, decay_rate=0.96, staircase=True)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy'])
        model.summary()

        self.model = model

    def train(self, x, y, epochs):
        # load data
        x = pickle.load(open(x, "rb"))
        y = pickle.load(open(y, "rb"))
        x = np.array(x)
        y = np.array(y)
        # tensorboard
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/fit/" + time
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # fit the model
        self.model.fit(x, y, batch_size=self.batch_size, epochs=epochs, validation_split=0.3, callbacks=[tensorboard_callback]) # %tensorboard --logdir logs/fit
        self.model.save(f"{time}_model")

    def evaluate(self, test_x, test_y): # test_ has to be pickle
        print("Evaluation:")


        result = self.model.evaluate(x, y)
        print(dict(zip(self.model.metrics_names, result)))

    def print_confusion_matrix(self, test_x, test_y): # test_ds has to be pickle
        print("Confusion matrix:")
        x = pickle.load(open(test_x, "rb"))
        y = pickle.load(open(test_y, "rb"))
        x = np.array(x)
        y = np.array(y)

        self.catagories = catagories
        predictions = []
        true_labels = []
        for photos, labels in self.test_pds:
            prediction = np.argmax(self.model.predict(photos), axis=-1)
            predictions = tf.concat([predictions, prediction], 0)
            true_labels = tf.concat([true_labels, labels], 0)

        matrix = tf.math.confusion_matrix(true_labels, predictions)

        cmap = sns.cubehelix_palette(as_cmap=True, reverse=False, light=1)
        ax = sns.heatmap(matrix, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap=cmap)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()

    def query(self, path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img, dtype=np.uint8)

        num_prediction = self.model.predict(tf.expand_dims(img_array, 0))
        num_label = np.argmax(num_prediction)
        label = class_names[num_label]

        score = tf.nn.softmax(num_prediction[0])
        confidence = 100* np.max(score)

        print(f"This image most likely belongs to {label} with a {confidence:.2f} percent confidence.")
        return (img_array, num_label)
