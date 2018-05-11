import glob
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import random
from socket import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

print("Creating Model")
model = load_model('Graph/LocalNet-Model.hdf5')

print(model.summary())
serverSocket = socket(AF_INET, SOCK_DGRAM)

# Assign IP address and port number to socket
serverSocket.bind(('', 7777))
print("Starting UDP Server on port: 7777" )

def shift(l, n):
    return l[n:] + l[:n]

result_memory_first = False
result_memory_len = 7
result_memory = np.zeros(shape=(result_memory_len,2)) # Array of last results used to reduce noise


    


prediction = [0,0]
while True:
    # Receive the client packet along with the address it is coming from
    message, address = serverSocket.recvfrom(1024)

    # Capitalize the message from the client
    message = message.decode('utf-8')
    words = message.split(",");
#    print("Got Message: " +str(len(words)))

    # Otherwise, the server responds

    # answer = 0.06281613,-0.8299413,

    headingText = words[0]
    lidarText = words[1:]

    heading = np.array([headingText])
    heading = heading.astype(np.float)


   
    
    try:
        lidar= np.array([lidarText])
        lidar = lidar.astype(np.float)

    
        lidar = np.expand_dims(lidar, axis=2)

        prediction = model.predict([lidar,heading], steps=1)[0]
   
        if result_memory_first is not True:
            print("Filling Trackable Memory")
            for x in range(0, result_memory_len):
                result_memory[x] = prediction
            result_memory_first = True
        else:
            result_memory = np.roll(result_memory, 1, axis=0)
            result_memory[0] = prediction 
           
          
      
        #prediction = np.sqrt(np.mean(result_memory**2), axis=0)
        prediction = result_memory.mean(axis=0)
        serverSocket.sendto(str(str(prediction[0]) + "," + str(prediction[1])).encode('utf-8'), address)
    except Exception:
        print("Error Running. Sending last result: " + str(Exception))
        #print("Raw Input: " + str(words))
        serverSocket.sendto(str(str(prediction[0]) + "," + str(prediction[1])).encode('utf-8'), address)
 #   print(prediction)
