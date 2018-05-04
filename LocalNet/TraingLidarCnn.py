import glob
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
print("Loading Data...");

def get_testing_data():
    path = 'Data/Test'  # use your path
    allFiles = glob.glob(path + "/*")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)
    return frame

def get_training_data():
    path = 'Data/Train'  # use your path
    allFiles = glob.glob(path + "/*")

    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)
    return frame

train_set = get_training_data()
test_set = get_testing_data()


x_train = train_set.drop(["X-Coord", "Y-Coord"], axis=1)
y_train = train_set[["X-Coord", "Y-Coord"]]

x_test = test_set.drop(["X-Coord", "Y-Coord"], axis=1)
y_test = test_set[["X-Coord", "Y-Coord"]]

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

model = Sequential()
model.add(Conv1D(32, kernel_size=5, strides=(1),
                 activation='relu',
                 input_shape=(61,1)))

model.add(AveragePooling1D(pool_size=(2), strides=(2)))
model.add(Conv1D(64, 5, activation='relu'))
model.add(AveragePooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.01),
              )
filepath = "./Graph/mnist-cnn-best.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',  write_graph=True, write_images=True, batch_size=128)

model.fit(x_train, y_train,
          batch_size=164,
          epochs=50,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint,tbCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
model.save_weights('my_model_weights.h5')
print('Test loss:', score)
