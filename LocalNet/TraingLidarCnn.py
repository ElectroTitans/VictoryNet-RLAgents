import glob
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, Input, Add, Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from google.cloud import datastore
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from pprint import pprint
import yaml
from shutil import copyfile
import os
import datetime

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

if not os.path.exists( "./Graph/"+model_name+"/"):
    os.makedirs( "./Graph/"+model_name+"/")

print("Backing Up Model Settings...")
copyfile("settings.yaml", "./Graph/"+model_name+"/settings.yaml")


datastore_client = datastore.Client('victory-net')

# The kind for the new entity
kind = 'Model'
# The name/ID for the new entity
name = model_name
# The Cloud Datastore key for the new entity
task_key = datastore_client.key(kind, name)

# Prepares the new entity
task = datastore.Entity(key=task_key)
task['info_date']  = datetime.datetime.now()
task['info_name']  = model_name
task['info_desc']  = cfg['model_desc']
task['info_status'] = "Loading Training Data"
task['cfg_model_version']   = cfg['model_version']
task['cfg_model_edit']      = cfg['model_edit']
task['cfg_model_run']       = cfg['model_run']
task['cfg_learning_rate']   = cfg['learning_rate']
task['cfg_batch_size']      = cfg['batch_size']
task['cfg_epoch']           = cfg['epoch']
task['cfg_conv1_filter']    = cfg['conv1_filter']
task['cfg_conv2_filter']    = cfg['conv2_filter']
task['cfg_conv1_kernal']     = cfg['conv1_kernal']
task['cfg_conv2_kernal']     = cfg['conv2_kernal']
task['cfg_fully_connected'] = cfg['fully_connected']


task['env_lineNum']      = env_settings['lineNum']
task['env_noise']        = env_settings['noise']
task['env_dropout']      = env_settings['dropout']
task['env_maxRange']     = env_settings['maxRange']
task['env_spinRate']     = env_settings['spinRate']
task['env_instantMode']  = env_settings['instantMode']


# Saves the entity
datastore_client.put(task)

print('Saved to GCP Datastore{}'.format(task.key))


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

task['info_training_length'] = len(train_set)
datastore_client.put(task)


x2_train = train_set[["Heading"]]
y_train = train_set[["X-Coord", "Y-Coord"]]
train_set.drop(["X-Coord", "Y-Coord", "Heading"], axis=1, inplace=True)
x1_train = np.expand_dims(train_set, axis=2)

print(x1_train.shape)
print("Loading Testing Data...")
test_set = get_testing_data()

task['info_testing_length'] = len(test_set)
datastore_client.put(task)
print('Updated to GCP Datastore{}'.format(task.key))

x2_test = test_set[["Heading"]]
y_test = test_set[["X-Coord", "Y-Coord"]]
test_set.drop(["X-Coord", "Y-Coord",  "Heading"], axis=1, inplace=True)
x1_test = np.expand_dims(test_set, axis=2)

filepath = "./Graph/"+model_name+"/trained_model.hdf5"

if cfg["load"] is True:
    
    print("Loading Previous Model from: " + filepath)
    model = load_model(filepath)
    print(model.summary())

else:
    print("Creating Model")
    lidar_input       = Input(shape=(env_settings['lineNum'],1), name='lidar_input')
    lidar_conv1       = Conv1D(cfg["conv1_filter"], kernel_size=cfg["conv1_kernal"], strides=(1),activation='relu', name='lidar_conv1')(lidar_input)
    lidar_pooling1    = AveragePooling1D(pool_size=(2), strides=(2),  name='lidar_pooling1')(lidar_conv1)
    lidar_conv2       = Conv1D(cfg["conv2_filter"],kernel_size=cfg["conv2_kernal"], activation='relu',  name='lidar_conv2')(lidar_pooling1)
    lidar_pooling2    = AveragePooling1D(pool_size=(2),  name='lidar_pooling2')(lidar_conv2)
    lidar_flatten     = Flatten( name='lidar_flatten')(lidar_pooling2)

    imu_input         = Input(shape=(1,) ,name='imu_input')
    combined_layer    = Concatenate(name='combined_layer')( [lidar_flatten, imu_input])

    final_dense       = Dense(cfg["fully_connected"], activation='relu', name='final_dense')(combined_layer)
    coord_out         = Dense(2, name='coord_out')(final_dense)



    model = Model(inputs=[lidar_input, imu_input], outputs=coord_out)

    print(model.summary())
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=cfg['learning_rate']))



scale = tf.placeholder(tf.float32)        

    # Note, `scaled` above is a tensor. Its being passed `draw_scatter` below. 
    # However, when `draw_scatter` is invoked, the tensor will be evaluated and a
    # numpy array representing its content is provided.   





checkpoint = ModelCheckpoint(filepath,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
embedding_layer_names = [ 'lidar_conv1']


class GCPDatastoreCheckpoint(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        task['info_epoch'] = epoch + 1
        task['info_status'] = "Training"
        task['info_progress'] = (epoch + 1) / cfg['epoch']
        task['info_loss'] = logs.get('loss')
        task['info_val_loss'] = logs.get('val_loss')
        task['info_losses'] = self.losses
        datastore_client.put(task)
        print('Updated to GCP Datastore{}'.format(task.key))



gcp_checkpoint = GCPDatastoreCheckpoint()
tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./Graph/'+model_name, 
    histogram_freq=5,
    write_grads=True,
    write_graph=True,
    batch_size=cfg["batch_size"])


task['info_status'] = "Training (first epoch)"
datastore_client.put(task)
print('Updated to GCP Datastore{}'.format(task.key))

model.fit([x1_train, x2_train], y_train,
          batch_size=cfg["batch_size"],
          epochs=cfg["epoch"],
          verbose=1,
          shuffle=True,
          validation_data=([x1_test, x2_test], y_test),
          callbacks=[gcp_checkpoint, checkpoint,tbCallBack])


score = model.evaluate([x1_test, x2_test], y_test, verbose=1)

print('Test loss:', score)
task['info_status'] = "Complete"
task['info_final_loss'] = score
datastore_client.put(task)
print('Updated to GCP Datastore{}'.format(task.key))