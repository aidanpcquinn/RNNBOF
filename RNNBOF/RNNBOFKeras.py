import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



# define keras RNNBOF model
    # inputs:
        # shape: input shape of a single window for a single entity
        # numLayers: number of stacked rnn lstm layers
        # dropout: float representing the percent of nodes per layer to be droped [0-1]
        # l2Val: l2 kernal reg value
        # learnRate: learning rate of the Adam learning algorithm
        # numHiddenNodes: number of hidden nodes
    # returns:
        # keras model


def get_lstm(shape, numLayers, dropout, l2Val, learnRate, numHiddenNodes):

    # defines  input shape of a single window for a single entity
    inputs = keras.Input(shape)
    
    # split for the 1 layer, 2 layer and 2> layer cases as last layer sets return_sequences = False and first layer has input as inputs variable
    if numLayers == 1:
        # keras LSTM layer, with l2 kernal reg
        x = keras.layers.LSTM(int(numHiddenNodes), kernel_regularizer=l2(l2Val))(inputs)
        x = keras.layers.Dropout(dropout)(x)
        
    elif numLayers == 2:
        x = keras.layers.LSTM(int(numHiddenNodes), return_sequences = True, kernel_regularizer=l2(l2Val))(inputs)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.LSTM(int(numHiddenNodes), kernel_regularizer=l2(l2Val))(x)
        x = keras.layers.Dropout(dropout)(x)
        
    else:
        x = keras.layers.LSTM(int(numHiddenNodes), return_sequences = True, kernel_regularizer=l2(l2Val))(inputs)
        x = keras.layers.Dropout(dropout)(x)
        
        for i in range(int(numLayers-2)):            
            x = keras.layers.LSTM(int(numHiddenNodes), return_sequences = True, kernel_regularizer=l2(l2Val))(x)
            x = keras.layers.Dropout(dropout)(x)

        x = keras.layers.LSTM(int(numHiddenNodes), kernel_regularizer=l2(l2Val))(x)
        x = keras.layers.Dropout(dropout)(x)

    # output sigmoid layer
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    # define model
    rnn_model = keras.Model(inputs, output)

    # compile model with loss function and optimizer
    rnn_model.compile(loss="binary_crossentropy", optimizer= Adam(lr = learnRate))

    return rnn_model