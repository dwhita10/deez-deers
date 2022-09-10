import cv2 as cv
import numpy as numpy
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten

im = cv.imread("Example image/no_deer.jpg", cv.IMREAD_GRAYSCALE)
print(im.shape)

input_images = [im]

input_shape = (1, im.shape[0], im.shape[1])
model = Sequential()
model.add(Conv2D(100, (50, 50), activation='relu', input_shape=(im.shape[0], im.shape[1], 1)))
model.add(Flatten())
model.add(Dense(50, 'sigmoid',input_shape=(100,)))
model.add(Dense(1))
model.compile(optimizer="adam", loss='binary_crossentropy')
model.summary()

