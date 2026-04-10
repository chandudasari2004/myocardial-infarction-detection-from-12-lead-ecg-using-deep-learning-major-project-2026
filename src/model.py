import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling1D, Dropout, Bidirectional, LSTM, Dense

def build_model(window_size, n_leads):
    model = Sequential()

    model.add(Conv1D(32, 5, padding="same", input_shape=(window_size, n_leads)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Conv1D(64, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
