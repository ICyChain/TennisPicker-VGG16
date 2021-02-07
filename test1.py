import tensorflow as tf
import cv2
import numpy as np
#
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#
model_path = "E:\\PycharmProjects\\Tensorflow\\IdentifyTennisTest01\\imgmodel_vgg16_2.h5"
model = tf.keras.models.load_model(model_path)
img = cv2.imread("E:\\PycharmProjects\\Tensorflow\\IdentifyTennisTest01\\img\\test\\Tennis\\3.jpg")

img = cv2.resize(img, (224, 224))
img = np.array(img, np.float32)
x = np.expand_dims(img, 0)
y = model.predict(x)

print("predict: ", y)