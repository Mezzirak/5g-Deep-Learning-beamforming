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
        
    Step-by-step explanation:
    1. Compute sample covariance matrix R from the received signal
    2. Add diagonal loading for numerical stability (prevents singular matrix)
    3. Extract steering vector for the desired user
    4. Solve R * w = a_d using stable linear solver (not direct inversion)
    5. Normalise the weights to satisfy distortionless constraint
    6. Apply weights to received signal to extract the desired user
    """
    num_symbols = received_signal.shape[1]
    
    # 1: compute sample covariance matrix
    # R = (1/N) * Y * Y^H where Y is received signal matrix
    Y = received_signal.T  # transpose to (num_symbols x num_antennas)
    R = (Y.conj().T @ Y) / num_symbols
    
    # 2: diagonal loading for numerical stability
    # This prevents matrix from being singular or ill-conditioned
    # Especially important at high SNR where R can become nearly singular
    R = R + diagonal_loading * np.eye(R.shape[0])
    
    # 3: xxtract steering vector for desired user
    a_d = channel_vectors[desired_user_idx, :]
    
    # 4: Solve R * w_numerator = a_d using stable solver
    # This is more stable than computing R_inv = inv(R) then R_inv @ a_d
    # np.linalg.solve uses LU decomposition which is numerically robust
    w_numerator = np.linalg.solve(R, a_d)
    
    # 5: Compute denominator for normalisation
    # This ensures unit gain in desired direction (distortionless constraint)
    denominator = a_d.conj().T @ w_numerator
    
    # 6: Final MVDR weights
    w_mvdr = w_numerator / denominator
    
    # 7: Apply beamformer weights to received signal
    # This performs spatial filtering to extract the desired signal
    estimated_symbols = w_mvdr.conj().T @ received_signal
    
    return estimated_symbols, w_mvdr


def create_dl_model(num_antennas, learning_rate=0.001, dropout_rate=0.2):
    """
    Creates an improved deep learning model for beamforming
    
    Architecture improvements over original:
    1. Deeper network (3 hidden layers instead of 2)
    2. Batch Normalisation for training stability
    3. Dropout for regularisation (prevents overfitting)
    4. L2 weight regularisation
    5. Wider layers (256 neurons instead of 128)
    
    The network learns to map:
        Input: Received signal at antenna array (complex, split into real/imag)
        Output: Transmitted symbol of desired user (complex, split into real/imag)
    
    Args:
        num_antennas: Number of antennas (determines input size)
        learning_rate: Adam optimiser learning rate
        dropout_rate: Fraction of neurons to drop during training
        
    Returns:
        Compiled Keras model
        
    Explanation of each layer:
    - Input: 2*num_antennas (real and imaginary parts concatenated)
    - Dense(256) + ReLU: First feature extraction layer
    - Batch Normalisation: Normalises activations, speeds up training
    - Dropout(0.2): Randomly drops 20% of neurons, prevents overfitting
    - Dense(256) + ReLU: Second feature extraction layer
    - Dense(128) + ReLU: Third feature extraction layer (deeper processing)
    - Dense(2) linear: Output layer (real and imag of symbol)
    """
    model = Sequential([
        # Input layer receives concatenated real and imaginary parts
        Input(shape=(num_antennas * 2,)),
        
        # First hidden layer for wide feature extraction
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),  # Normalize activations
        Dropout(dropout_rate),  # Regularization
        
        # Second hidden layer
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Third hidden layer for deeper processing
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer, 2 neurons for real and imaginary parts
        Dense(2, activation='linear')
    ])
    
    # Compile with Adam optimiser and MSE loss
    # MSE (Mean Squared Error) measures average squared difference between
    # predicted and actual symbols
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def prepare_dl_data(received_signal, transmitted_symbols, desired_user_idx):
    """
    Prepares data for training/testing the DL model
    
    Converts complex-valued signals to real-valued format by splitting
    into real and imaginary components
    
    Args:
        received_signal: (num_antennas x num_symbols) complex array
        transmitted_symbols: (num_users x num_symbols) complex array
        desired_user_idx: Which user's symbols to predict
        
    Returns:
        X: Input features (num_symbols x 2*num_antennas)
        Y: Target labels (num_symbols x 2)
        
    Why we split complex into real/imaginary:
    Neural networks expect real-valued inputs, but our signals are complex
    By concatenating [real_part, imag_part], we preserve all information
    while making it compatible with standard neural network frameworks
    """
    # Convert received signal from (antennas x symbols) to (symbols x antennas)
    X_complex = received_signal.T
    
    # Split into real and imaginary, then concatenate horizontally
    # Shape: (num_symbols, 2*num_antennas)
    X = np.hstack([np.real(X_complex), np.imag(X_complex)])
    
    # Extract the desired user's symbols and convert to (samples x 2) format
    Y_complex = transmitted_symbols[desired_user_idx, :].T
    Y = np.column_stack([np.real(Y_complex), np.imag(Y_complex)])
    
    return X, Y