from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.initializers import Constant

def make_cnn(input_shape, num_classes, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    if num_classes==2:
        output_layer = keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(gap)
    else:
        output_layer = keras.layers.Dense(num_classes, activation="softmax", bias_initializer=output_bias)(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def make_lstm(input_shape, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    lstm_model = Sequential()
    lstm_model.add(Bidirectional(LSTM(20, activation='relu'), input_shape=input_shape))
    lstm_model.add(Dense(30, activation='relu'))
    lstm_model.add(Dense(10, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid',bias_initializer=output_bias))
    return lstm_model