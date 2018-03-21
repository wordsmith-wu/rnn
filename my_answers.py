import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[ii:ii+window_size] for ii in range(len(series[:-window_size]))]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    # layer 1: LSTM module with 5 hidden units
    model.add(LSTM(5,input_shape=(window_size, 1)))
    # layer 2: fully connected module with one unit
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    lowercase = [chr(n) for n in range(97,123)]
    space = [' ',]
    text = ''.join([c for c in text if c in punctuation+lowercase+space])

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[ii:ii+window_size] for ii in range(0,len(text[:-window_size]),step_size)]
    outputs = [text[ii] for ii in range(window_size, len(text),step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    # layer 1: LSTM module with 200 hidden units
    model.add(LSTM(200,input_shape=(window_size,num_chars)))
    # layer 2: linear module, fully connected, with len(chars) hidden units
    model.add(Dense(num_chars))
    # layer 3: softmax activation
    model.add(keras.layers.Activation('softmax'))
    return model
