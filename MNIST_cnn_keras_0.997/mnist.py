import os
import tensorflow as tf
import keras.optimizers as method
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
import scipy.io as sio
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_dict = sio.loadmat('MNIST.mat')
samples = data_dict['fea']
labels = data_dict['gnd']
x = samples
x = x.reshape(x.shape[0], 28, 28, 1).astype('float32')
x /= 255

y = labels
y = np_utils.to_categorical(y)
train_x = x[:60000]
test_x = x[60000:]
train_y = y[:60000]
test_y = y[60000:]

s = int(train_x.shape[0]/2)
train_x1 = train_x[:s]
train_x2 = train_x[s:]


train_y1 = train_y[:s]
train_y2 = train_y[s:]



def my_model():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(7, 7), padding='same', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model









model = my_model()
optimizer = method.RMSprop(lr=0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
h1 = model.fit(train_x1, train_y1, validation_data=(test_x, test_y), epochs=10, batch_size=200, verbose=2)

optimizer = method.RMSprop(lr=0.0001)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
h2 = model.fit(train_x2, train_y2, validation_data=(test_x, test_y), epochs=10, batch_size=200, verbose=2)

optimizer = method.RMSprop(lr=0.00001)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
h3 = model.fit(train_x1, train_y1, validation_data=(test_x, test_y), epochs=10, batch_size=200, verbose=2)

optimizer = method.RMSprop(lr=0.00001)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
h4 = model.fit(train_x2, train_y2, validation_data=(test_x, test_y), epochs=10, batch_size=200, verbose=2)


scores = model.evaluate(test_x, test_y, verbose=0)
print(scores[1]*100)

#plot
ax = []
ax.append(h1.history['acc'])
ax.append(h2.history['acc'])
ax.append(h3.history['acc'])
ax.append(h4.history['acc'])
ax1 = np.reshape(ax, (1,40))
ax2 = ax1[0]
bx = []
bx.append(h1.history['val_acc'])
bx.append(h2.history['val_acc'])
bx.append(h3.history['val_acc'])
bx.append(h4.history['val_acc'])
bx1 = np.reshape(ax, (1,40))
bx2 = bx1[0]

plt.plot(ax2)
plt.plot(bx2)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


cx = []
cx.append(h1.history['loss'])
cx.append(h2.history['loss'])
cx.append(h3.history['loss'])
cx.append(h4.history['loss'])
cx1 = np.reshape(cx, (1,40))
cx2 = cx1[0]
dx = []
dx.append(h1.history['val_loss'])
dx.append(h2.history['val_loss'])
dx.append(h3.history['val_loss'])
dx.append(h4.history['val_loss'])
dx1 = np.reshape(dx, (1,40))
dx2 = dx1[0]

plt.plot(cx2)
plt.plot(dx2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

model_json = model.to_json()
with open("model1.json", "w") as json_file :
    json_file.write(model_json)
model.save_weights("model1.h5")










