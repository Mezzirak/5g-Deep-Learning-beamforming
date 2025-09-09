import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def mvdr_beamformer(received_signal, channel_vectors, desired_user_idx):
    """
    Performs MVDR beamforming.
    """
    num_symbols = received_signal.shape[1]
    Y = received_signal.T 
    R = (Y.conj().T @ Y) / num_symbols
    
    a_d = channel_vectors[desired_user_idx, :]
    R_inv = np.linalg.inv(R)
    
    numerator = R_inv @ a_d
    denominator = a_d.conj().T @ R_inv @ a_d
    w_mvdr = numerator / denominator
    
    estimated_symbols = w_mvdr.conj().T @ received_signal
    return estimated_symbols, w_mvdr

def create_dl_model(num_antennas):
    """
    Creates the deep learning model architecture.
    """
    model = Sequential([
        Input(shape=(num_antennas * 2,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_dl_data(received_signal, transmitted_symbols, desired_user_idx):
    """
    Prepares data for training the DL model.
    """
    X_complex = received_signal.T
    X_train = np.hstack([np.real(X_complex), np.imag(X_complex)])
    
    Y_complex = transmitted_symbols[desired_user_idx, :].T
    Y_train = np.column_stack([np.real(Y_complex), np.imag(Y_complex)])
    
    return X_train, Y_train