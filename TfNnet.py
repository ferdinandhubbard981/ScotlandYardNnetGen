import sys
from utils import *

import tensorflow as tf
# boardSize, actionSize, batchSize, numChannels, dropout, lr, 
class TfNnet():
    def __init__(self, boardSize, actionSize, batchSize, numOfPlayers, numChannels, dropout, lr):
        # game params
        self.action_size = actionSize
        self.lr = lr

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.input_boards = tf.placeholder(tf.float32, shape=[boardSize * numOfPlayers], name="input")    # s: batch_size x board_x x board_y
        print("self.input_boards: ", self.input_boards)
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        print("self.dropout: ", self.dropout)
        self.isTraining = tf.placeholder(tf.bool, name="is_training")

        x_image = tf.reshape(self.input_boards, [-1, boardSize, numOfPlayers])#change shape for batch size               # batch_size  x board_x x board_y x 1
        print("x_image: ", x_image)
        h_conv1 = Relu(BatchNormalization(self.conv1d(x_image, numChannels, 'same'), axis=2, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
        print("h_conv1: ", h_conv1)
        h_conv2 = Relu(BatchNormalization(self.conv1d(h_conv1, numChannels, 'same'), axis=2, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
        print("h_conv2: ", h_conv2)
        h_conv3 = Relu(BatchNormalization(self.conv1d(h_conv2, numChannels, 'valid'), axis=2, training=self.isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
        print("h_conv3: ", h_conv3)
        h_conv4 = Relu(BatchNormalization(self.conv1d(h_conv3, numChannels, 'valid'), axis=2, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
        print("h_conv4: ", h_conv4)
        h_conv4_flat = tf.reshape(h_conv4, [-1, numChannels * (boardSize-4)])
        print("h_conv4_flat: ", h_conv4_flat)
        s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024, use_bias=False), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
        # s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024, use_bias=False), axis=1)), rate=self.dropout) # batch_size x 1024
        print("s_fc1: ", s_fc1)
        s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512, use_bias=False), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
        # s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512, use_bias=False), axis=1)), rate=self.dropout)         # batch_size x 512
        print("s_fc2: ", s_fc2)
        self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
        print("self.pi: ", self.pi)
        self.prob = tf.nn.softmax(self.pi, name="policy_output")
        print("self.prob: ", self.prob)
        self.v = Tanh(Dense(s_fc2, 1), name="state_value_output")                                                               # batch_size x 1
        print("self.v: ", self.v)

        self.calculate_loss()
        init = tf.global_variables_initializer()
        saver_def = tf.train.Saver().as_saver_def()
        
        print('Operation to initialize variables:       ', init.name)
        print('Tensor to feed as input data:            ', self.input_boards.name)
        print('Tensor to feed as dropout val:            ', self.dropout.name)
        print('Tensor to feed as training policy:      ', self.target_pis.name)
        print('Tensor to feed as is_training:      ', self.isTraining.name)
        print('Tensor to feed as training value:      ', self.target_vs.name)
        print('Tensor to fetch as policy prediction:           ', self.prob.name)
        print('Tensor to fetch as val prediction:           ', self.v.name)
        print('Tensor to fetch as loss_pi:           ', self.loss_pi.name)
        print('Tensor to fetch as total_loss:           ', self.total_loss.name)
        print('Operation to train one step:             ', self.train_step.name)
        print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
        print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
        print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
        
        with open('graph.pb', 'wb') as f:
            f.write(tf.get_default_graph().as_graph_def().SerializeToString())
            
    def conv1d(self, x, out_channels, padding):
      return tf.layers.conv1d(x, out_channels, kernel_size=[3], padding=padding, use_bias=False)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[1, self.action_size], name="targetPolicy")
        print("self.target_pis: ", self.target_pis)
        self.target_vs = tf.placeholder(tf.float32, shape=[1], name="targetV")
        print("self.target_vs: ", self.target_vs)
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        print("self.loss_pi: ", self.loss_pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        print("self.loss_v: ", self.loss_v)
        self.total_loss = self.loss_pi + self.loss_v
        print("self.total_loss: ", self.total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)