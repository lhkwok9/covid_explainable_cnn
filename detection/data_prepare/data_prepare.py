import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

class DataPrepare:
    def __init__(self, data_dir, catagories, img_width: int, img_height: int):
        self.data_dir = data_dir
        self.catagories = catagories
        self.img_width = img_width
        self.img_height = img_height
        self.training_data = []
        self.data_pickle = []
        self.label_pickle = []

    def showExample(self, resize: bool):
        for category in self.catagories:
            path = os.path.join(self.data_dir, category)  # create path
            for img in os.listdir(path):  # iterate over each image per catagories
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array  #cv2.IMREAD_COLOR/cv2.IMREAD_GRAYSCALE

                if resize:
                    img_array = cv2.resize(img_array, (self.img_width, self.img_height))
                cv2.imshow("image", img_array)  # display!
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                break  # we just want one for now so break
            break  #...and one more!

    def make_train_validate(self, x_name, y_name, grey_scale: bool): # .pickle
        for category in self.catagories:
            path = os.path.join(self.data_dir, category, "")
            class_num = self.catagories.index(category)
            for img in os.listdir(path):
                if grey_scale:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                else:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (self.img_width, self.img_height))
                self.training_data.append([img_array, class_num]) # [[img, label], [img, label], ...]

        random.shuffle(self.training_data)
        for features, label in self.training_data:
            self.data_pickle.append(features)
            self.label_pickle.append(label)
        if grey_scale:
            self.data_pickle = np.array(self.data_pickle).reshape(-1, self.img_width, self.img_height, 1) # 1 if grey scale, 3 if color
        else:
            self.data_pickle = np.array(self.data_pickle).reshape(-1, self.img_width, self.img_height, 3)

        pickle_out = open(x_name, "wb")
        pickle.dump(self.data_pickle, pickle_out)
        pickle_out.close()

        pickle_out = open(y_name, "wb")
        pickle.dump(self.label_pickle, pickle_out)
        pickle_out.close()
        print("Finish pickle")
