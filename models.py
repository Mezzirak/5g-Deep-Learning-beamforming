import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def mvdr_beamformer(received_signal, channel_vectors, desired_user_idx, 
                   diagonal_loading=1e-6):
    """
    Performs MVDR (Minimum Variance Distortionless Response) beamforming
    
    Mathematical formula: w = (R^-1 * a_d) / (a_d^H * R^-1 * a_d)
    Where:
        R = covariance matrix of received signal
        a_d = steering vector of desired user
        w = optimal weight vector
    
    The MVDR beamformer minimises output power (variance) while maintaining
    unit gain in the direction of the desired user
    
    Args:
        received_signal: (num_antennas x num_symbols) received signal matrix
        channel_vectors: (num_users x num_antennas) channel matrix
        desired_user_idx: Index of the desired user
        diagonal_loading: Regularisation for numerical stability
        
    Returns:
        estimated_symbols: Recovered symbols for desired user
        w_mvdr: Computed weight vector
    """
    num_symbols = received_signal.shape[1]
    
    # 1: compute sample covariance matrix
    # R = (1/N) * Y * Y^H where Y is received signal matrix
    Y = received_signal.T  # transpose to (num_symbols x num_antennas)
    R = (Y.conj().T @ Y) / num_symbols
    
    # 2: diagonal loading for numerical stability
    R = R + diagonal_loading * np.eye(R.shape[0])
    
    # 3: extract steering vector for desired user
    a_d = channel_vectors[desired_user_idx, :]
    
    # 4: Solve R * w_numerator = a_d using stable solver
    w_numerator = np.linalg.solve(R, a_d)
    
    # 5: Compute denominator for normalisation
    denominator = a_d.conj().T @ w_numerator
    
    # 6: Final MVDR weights
    w_mvdr = w_numerator / denominator
    
    # 7: Apply beamformer weights to received signal
    estimated_symbols = w_mvdr.conj().T @ received_signal
    
    return estimated_symbols, w_mvdr


def create_dl_model(num_antennas, learning_rate=0.001, dropout_rate=0.2):
    """
    Creates the deep learning model for beamforming.
    
    The network learns to map:
        Input: Received signal + Steering vector (Real + Imaginary parts for both)
        Output: Transmitted symbol of desired user
    
    Input now includes steering vector
    16 antennas * 4 (received_real, received_imag, steering_real, steering_imag) = 64 features
    """
    # Calculate input shape: MUST include steering vector
    input_dim = num_antennas * 4  # CHANGED FROM * 2 TO * 4
    
    model = Sequential([
        # Input layer
        Input(shape=(input_dim,)),
        
        # First hidden layer
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second hidden layer
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Third hidden layer
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer (Real and Imag parts of the symbol)
        Dense(2, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def prepare_dl_data(received_signal, transmitted_symbols, channel_vectors, desired_user_idx):
    """
    Prepares data for training/testing the DL model WITH steering vector.
    
    include the steering vector so the model knows
    which direction to look for the desired user!
    
    Args:
        received_signal: (num_antennas x num_symbols) complex array
        transmitted_symbols: (num_users x num_symbols) complex array
        channel_vectors: (num_users x num_antennas) complex array - ADDED!
        desired_user_idx: Which user's symbols to predict
    
    Returns:
        X: Input features (num_symbols x 4*num_antennas) - includes steering vector
        Y: Target labels (num_symbols x 2)
    """
    # Convert received signal from (antennas x symbols) to (symbols x antennas)
    X_received = received_signal.T
    
    # Get the steering vector for the desired user
    steering_vector = channel_vectors[desired_user_idx, :]
    
    # Tile the steering vector to match each symbol (repeat for all symbols)
    num_symbols = X_received.shape[0]
    X_steering = np.tile(steering_vector, (num_symbols, 1))
    
    # Concatenate: [received_real, received_imag, steering_real, steering_imag]
    # This gives the model both the mixed signal and which direction to look!
    X = np.hstack([
        np.real(X_received),    # 16 values (received signal real part)
        np.imag(X_received),    # 16 values (received signal imaginary part)
        np.real(X_steering),    # 16 values (steering vector real part)
        np.imag(X_steering)     # 16 values (steering vector imaginary part)
    ])  # Total: 64 values
    
    # Extract the desired user's symbols if provided (for training)
    if transmitted_symbols is not None:
        Y_complex = transmitted_symbols[desired_user_idx, :].T
        Y = np.column_stack([np.real(Y_complex), np.imag(Y_complex)])
        return X, Y
    else:
        # For inference where we might not have labels
        return X, None