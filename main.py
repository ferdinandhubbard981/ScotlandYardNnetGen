import KerasNnet
import TfNnet
import tensorflow as tf
import numpy as np
def main():
    numOfPlayers = 6
    batchSize = 32
    boardSize = 199
    actionSize = 21778
    lr = 0.001
    dropout = 0.3
    epochs=10
    numChannels = 64
    nnet = TfNnet.TfNnet(boardSize, actionSize, batchSize, numOfPlayers, numChannels, dropout, lr)
    # nnet = KerasNnet.KerasNnet(boardSize, actionSize, batchSize, numOfPlayers, numChannels, dropout, lr)
    # nnet.model.summary()
    # input = np.random.uniform(low=0, high=1, size=(boardSize*numOfPlayers,))
    # input = tf.reshape(input, [1, boardSize*numOfPlayers])
    # output = nnet.model.predict(input, steps=1)
    # print("output size: " + str(len(output[0][0])))
    # tf.keras.models.save_model(nnet.model, "saved_model", save_format="tf")
    
    #builder = tf.saved_model.builder.SavedModelBuilder("./model_keras")
    
    # sess = tf.Session(graph=nnet.graph)
    # tf.io.write_graph(sess.graph_def, "myModel", "modelWeights.pb", False)
    
if __name__ == "__main__":
    main()
    print("working")
    