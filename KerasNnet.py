import sys
from utils import *

import argparse
# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

# boardSize, actionSize, batchSize, numChannels, dropout, lr, 
class KerasNnet():
    def __init__(self, boardSize, actionSize, batchSize, numOfPlayers, numChannels, dropout, lr):
        self.action_size = actionSize
        # Neural Net
        self.input_boards = Input(shape=(boardSize * numOfPlayers), name="input")    # s: batch_size x board_x x board_y
        # self.input_boards = tf.placeholder(tf.int32, shape=(None, boardSize * numOfPlayers), name="input")
        x_image = Reshape((boardSize,  numOfPlayers))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=2)(Conv1D(numChannels, 2, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=2)(Conv1D(numChannels, 2, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=2)(Conv1D(numChannels, 2, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=2)(Conv1D(numChannels, 2, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='policy_output')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='value_output')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(lr))

