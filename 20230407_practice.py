# import important packages
import tensorflow as tf
from tensorflow.keras import datasets, models, layers, utils, activations
from tensorflow.keras import losses, optimizers, metrics
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
''''''
# Load data (PS: Keras knows where to download. Don’t worry about it.)
if os.path.exists('my_mnist.npz'):
    my_mnist_data = np.load('my_mnist.npz')
    my_x_train = my_mnist_data['x_train']
    my_y_train = my_mnist_data['y_train']
    my_x_valid = my_mnist_data['x_valid']
    my_y_valid = my_mnist_data['y_valid']
else:
    # Save the data in file so that next time you can load from file
    # as in the midterm you will not have Internet
    (my_x_train, my_y_train), (my_x_valid, my_y_valid) = \
        datasets.mnist.load_data()
    np.savez_compressed('my_mnist', 
                        x_train=my_x_train, 
                        y_train=my_y_train,
                        x_valid=my_x_valid, 
                        y_valid=my_y_valid)
    
print(type(my_x_train))
print(my_x_train.shape)
img1 = my_x_train[0, :, :]
print(img1.shape)
'''
plt.imshow(img1)
plt.show()
plt.imshow(img1, cmap="gray")
plt.show
'''
# reshape image data by adding a dimension of channel
my_x_train_c1 = my_x_train.reshape(60000, 28, 28, 1)
my_x_valid_c1 = my_x_valid.reshape(10000, 28, 28, 1)

# categorical
my_y_train_categ = utils.to_categorical(my_y_train, 10)
my_y_valid_categ = utils.to_categorical(my_y_valid, 10)

# build a Convolutional Neural Network
my_model = models.Sequential()
my_model.add(layers.Input((28, 28, 1)))
my_model.add(layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(layers.GlobalAveragePooling2D())
my_model.add(layers.Dense(10, activation='softmax'))
my_model.compile(loss=losses.categorical_crossentropy, \
              optimizer = optimizers.SGD(),metrics = ['accuracy'])

# summarize the model. These statements are optional. 
# If your environment cannot plot model, 
# you can skip the statement.
# plot_model(my_model)
my_model.summary()

# Train the model
my_epochs = 1
my_logs = my_model.fit(my_x_train_c1, my_y_train_categ, batch_size = 128, \
                 epochs = my_epochs, verbose = 1, \
                 validation_data= (my_x_valid_c1, my_y_valid_categ))

# Save
# You can load your model by running
#   my_model = models.load_model('my_model_mnist_cnn_v1.h5')
my_model.save('my_model_mnist_cnn_v1.h5')

my_model = models.load_model('my_model_mnist_cnn_v1.h5')
# Predict your writing
my_handwriting_c1 = np.zeros((3, 28, 28, 1), dtype=np.uint8)
for i in range(3):
    my_handwriting_c1[i] = cv.imread('my_handwriting_%d.png' % i, cv.IMREAD_GRAYSCALE).reshape(28, 28, 1)
my_handwriting_prediction = my_model.predict(my_handwriting_c1)

# print the predictions of the first 3 images in test set
with np.printoptions(precision=2, suppress=True):
    print(my_handwriting_prediction)
''''''