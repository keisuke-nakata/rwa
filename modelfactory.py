from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.recurrent import LSTM

from rwa import RWA


def slice_last_timedimension(x):
    return x[:, -1, :]


def build_seq_length_model(units, batch_size, rwa=True):
    inp = Input(batch_shape=(batch_size, None, 1))  # (batch_size, time, vec)
    # if you don't need intermediate hidden states, specify `return_sequences=False` and omit `last_output` layer.
    if rwa:
        x = RWA(units=units, return_sequences=True)(inp)  # (batch_size, time, units)
    else:
        x = LSTM(units=units, return_sequences=True)(inp)
    last_output = Lambda(slice_last_timedimension)(x)  # (batch_size, 1, units)
    dense = Dense(1, activation='sigmoid')(last_output)  # (batch_size, 1)

    model = Model(inputs=[inp], outputs=[dense])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model


def build_addition_model(units, batch_size, rwa=True):
    inp = Input(batch_shape=(batch_size, None, 2))  # (batch_size, time, vec)
    # if you don't need intermediate hidden states, specify `return_sequences=False` and omit `last_output` layer.
    if rwa:
        x = RWA(units=units, return_sequences=True)(inp)  # (batch_size, time, units)
    else:
        x = LSTM(units=units, return_sequences=True)(inp)
    last_output = Lambda(slice_last_timedimension)(x)  # (batch_size, 1, units)
    dense = Dense(1)(last_output)  # (batch_size, 1)

    model = Model(inputs=[inp], outputs=[dense])
    model.compile(optimizer='adam', loss='mse')
    return model
