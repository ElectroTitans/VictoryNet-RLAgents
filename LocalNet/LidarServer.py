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
print("Creating Model")
model = load_model('Graph/mnist-cnn-best.hdf5')


serverSocket = socket(AF_INET, SOCK_DGRAM)

# Assign IP address and port number to socket
serverSocket.bind(('', 7777))
print("Starting UDP Server on port: 7777" )
while True:
    # Receive the client packet along with the address it is coming from
    message, address = serverSocket.recvfrom(1024)

    # Capitalize the message from the client
    message = message.decode('utf-8')
    words = message.split(",");
#    print("Got Message: " +str(len(words)))

    # Otherwise, the server responds

    # answer = 0.06281613,-0.8299413,

    mock_data = np.array([words])
    mock_data = mock_data.astype(np.float)
    mock_data =  np.expand_dims(mock_data, axis=2)

    prediction = model.predict(mock_data)
    serverSocket.sendto(str(str(prediction[0][0]) + "," + str(prediction[0][1])).encode('utf-8'), address)
 #   print(prediction)
