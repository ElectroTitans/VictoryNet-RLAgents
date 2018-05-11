import glob
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, Input, Add, Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from pprint import pprint
import yaml
print("Loading Model Settings...")


with open("settings.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
pprint(cfg)

model_name = "LocalNet " + str(cfg["model_version"]) + "_" + str(cfg["model_edit"]) + "_" + str( cfg["model_run"])
print("MODEL: " + model_name)
print("Loading Env Settings...")

with open('Data/_settings.json') as f:
    env_settings = json.load(f)
pprint(env_settings)
print("Input Lidar Length: " + str(env_settings['lineNum']))


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


x1_train = train_set.drop(["X-Coord", "Y-Coord", "Heading"], axis=1)
x2_train = train_set[["Heading"]]
y_train = train_set[["X-Coord", "Y-Coord"]]

x1_test = test_set.drop(["X-Coord", "Y-Coord",  "Heading"], axis=1)
x2_test = test_set[["Heading"]]
y_test = test_set[["X-Coord", "Y-Coord"]]

x1_train = np.expand_dims(x1_train, axis=2)
x1_test = np.expand_dims(x1_test, axis=2)




lidar_input       = Input(shape=(env_settings['lineNum'],1), name='lidar_input')
lidar_conv1       = Conv1D(8, kernel_size=3, strides=(1),activation='relu', name='lidar_conv1')(lidar_input)
lidar_pooling1    = AveragePooling1D(pool_size=(2), strides=(2),  name='lidar_pooling1')(lidar_conv1)
lidar_conv2       = Conv1D(16,kernel_size=3, activation='relu',  name='lidar_conv2')(lidar_pooling1)
lidar_pooling2    = AveragePooling1D(pool_size=(2),  name='lidar_pooling2')(lidar_conv2)
lidar_flatten     = Flatten( name='lidar_flatten')(lidar_pooling2)

imu_input         = Input(shape=(1,) ,name='imu_input')
combined_layer    = Concatenate(name='combined_layer')( [lidar_flatten, imu_input])

final_dense       = Dense(128, activation='relu', name='final_dense')(combined_layer)
coord_out         = Dense(2, name='coord_out')(final_dense)



model = Model(inputs=[lidar_input, imu_input], outputs=coord_out)

print(model.summary())

scale = tf.placeholder(tf.float32)        

    # Note, `scaled` above is a tensor. Its being passed `draw_scatter` below. 
    # However, when `draw_scatter` is invoked, the tensor will be evaluated and a
    # numpy array representing its content is provided.   
with tf.Session() as sess:
    env_arr = np.array([
        ["lineNum"], [env_settings["lineNum"]],
        ["noise"], [env_settings["noise"]],
        ["dropout"], [env_settings["dropout"]],
        ["maxRange"], [env_settings["maxRange"]],
        ["spinRate"], [env_settings["spinRate"]],
        ["instantMode"], [env_settings["instantMode"]]
    ]).reshape((6,2))
    cfg_arr = np.array([
        ["model_version"], [cfg["model_version"]],
        ["model_edit"], [cfg["model_edit"]],
        ["model_run"], [cfg["model_run"]],
        ["learning_rate"], [cfg["learning_rate"]],
        ["batch_size"], [cfg["batch_size"]],
        ["epoch"], [cfg["epoch"]]
    ]).reshape((6,2))
    env_summary = tf.summary.text('EnvSetting', tf.convert_to_tensor(env_arr))
    cfg_summary = tf.summary.text('ModelSettings', tf.convert_to_tensor(cfg_arr))

    all_summaries = tf.summary.merge_all() 

    writer = tf.summary.FileWriter('./Graph/'+model_name, sess.graph)
    summary = sess.run(all_summaries, feed_dict={scale: 2.})
    writer.add_summary(summary, global_step=0)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=cfg['learning_rate']))
              
filepath = "./Graph/LocalNet-"+model_name+".hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
embedding_layer_names = [ 'lidar_conv1']
tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./Graph/'+model_name, 
    histogram_freq=5,
    write_grads=True, 
    write_graph=True, 
    batch_size=cfg["batch_size"])

model.fit([x1_train, x2_train], y_train,
          batch_size=cfg["batch_size"],
          epochs=cfg["epoch"],
          verbose=1,
          shuffle=True,
          validation_data=([x1_test, x2_test], y_test),
          callbacks=[checkpoint,tbCallBack])
score = model.evaluate([x1_test, x2_test], y_test, verbose=0)

print('Test loss:', score)
