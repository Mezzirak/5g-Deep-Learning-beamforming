import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, Flatten, Reshape
from tensorflow.keras.regularizers import l2


def add_channel_estimation_error(channel_vectors, error_variance=0.1):
    """
    Add realistic channel estimation errors to simulate imperfect CSI.
    
    In real systems, channel estimation is never perfect due to:
    - Pilot contamination
    - Noise in channel estimation
    - Outdated CSI (user mobility)
    
    Args:
        channel_vectors: (num_users x num_antennas) perfect channel matrix
        error_variance: Variance of estimation error (0 = perfect, 0.1 = realistic)
        
    Returns:
        estimated_channels: Noisy version of channel_vectors
        
    Explanation:
        Error variance of 0.1 means roughly 10% error in channel magnitude
        Typical values: 0.01 (good), 0.1 (realistic), 0.3 (poor)
    """
    # Generate complex Gaussian noise
    noise_real = np.sqrt(error_variance / 2) * np.random.randn(*channel_vectors.shape)
    noise_imag = np.sqrt(error_variance / 2) * np.random.randn(*channel_vectors.shape)
    noise = noise_real + 1j * noise_imag
    
    # Add noise to perfect channels
    estimated_channels = channel_vectors + noise
    
    return estimated_channels


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


def zero_forcing_beamformer(received_signal, channel_vectors, desired_user_idx):
    """
    Zero-Forcing beamformer - completely nulls interference.
    
    Simpler than MVDR but can amplify noise at low SNR.
    
    Args:
        received_signal: (num_antennas x num_symbols) received signal matrix
        channel_vectors: (num_users x num_antennas) channel matrix
        desired_user_idx: Index of the desired user
        
    Returns:
        estimated_symbols: Recovered symbols for desired user
        w_zf: Computed weight vector
    """
    # Channel matrix: H = [h_0, h_1, ..., h_{K-1}]^T
    H = channel_vectors.T  # (num_antennas x num_users)
    
    # Pseudo-inverse of H
    W = np.linalg.pinv(H)  # (num_users x num_antennas)
    
    # Extract weights for desired user
    w_zf = W[:, desired_user_idx]  # (num_antennas,)
    
    # Apply weights
    estimated_symbols = w_zf.conj().T @ received_signal
    
    return estimated_symbols, w_zf


def create_dl_model_feedforward(num_antennas, learning_rate=0.001, dropout_rate=0.2):
    """
    Original feedforward architecture (your current model).
    
    Architecture: 256 → 256 → 128 → 2
    """
    input_dim = num_antennas * 4  # Includes steering vectors
    
    model = Sequential([
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
    ], name='FeedForward')
    
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimiser=optimiser, loss='mse', metrics=['mae'])
    
    return model


def create_dl_model_cnn(num_antennas, learning_rate=0.001, dropout_rate=0.2):
    """
    NEW: CNN-based architecture that exploits spatial structure.
    
    Key Idea:
    - Antenna array has spatial structure (nearby antennas see correlated signals)
    - Conv1D can learn spatial patterns across antennas
    - Better than fully-connected for array processing
    
    Architecture:
    1. Reshape input to (num_antennas, 4) treating antennas as spatial dimension
    2. Apply 1D convolutions across antennas
    3. Flatten and dense layers for final prediction
    
    Args:
        num_antennas: Number of antennas (16)
        learning_rate: Optimiser learning rate
        dropout_rate: Dropout probability
        
    Returns:
        Compiled Keras model
    """
    # Input: 64 features = (16 antennas × 4 features per antenna)
    # Features per antenna: [received_real, received_imag, steering_real, steering_imag]
    inputs = Input(shape=(num_antennas * 4,), name='input')
    
    # Reshape to (num_antennas, 4) to treat as spatial sequence
    x = Reshape((num_antennas, 4))(inputs)
    
    # First Conv1D layer: Learn local spatial patterns
    # Kernel size 3 means each filter looks at 3 adjacent antennas
    x = Conv1D(filters=64, kernel_size=3, activation='relu', 
               padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second Conv1D layer: Learn higher-level patterns
    x = Conv1D(filters=128, kernel_size=3, activation='relu', 
               padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Third Conv1D layer: Further spatial feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu', 
               padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Flatten to connect to dense layers
    x = Flatten()(x)
    
    # Dense layers for final processing
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output: 2 values (real and imaginary parts of symbol)
    outputs = Dense(2, activation='linear', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CNN')
    
    # Compile
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimiser=optimiser, loss='mse', metrics=['mae'])
    
    return model


def create_dl_model_deep(num_antennas, learning_rate=0.001, dropout_rate=0.2):
    """
    NEW: Deeper feedforward network with wider layers.
    
    Architecture: 512 → 256 → 256 → 128 → 64 → 2
    
    More capacity to learn complex patterns, but slower to train.
    """
    input_dim = num_antennas * 4
    
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Layer 1: Very wide
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 2
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 3
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 4
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 5: Narrow
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output
        Dense(2, activation='linear')
    ], name='DeepFeedForward')
    
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimiser=optimiser, loss='mse', metrics=['mae'])
    
    return model


# Alias for backwards compatibility
def create_dl_model(num_antennas, learning_rate=0.001, dropout_rate=0.2, architecture='feedforward'):
    """
    Factory function to create different DL architectures.
    
    Args:
        num_antennas: Number of antennas
        learning_rate: Optimiser learning rate
        dropout_rate: Dropout probability
        architecture: 'feedforward' (default), 'cnn', or 'deep'
        
    Returns:
        Compiled Keras model
    """
    if architecture == 'cnn':
        return create_dl_model_cnn(num_antennas, learning_rate, dropout_rate)
    elif architecture == 'deep':
        return create_dl_model_deep(num_antennas, learning_rate, dropout_rate)
    else:  # 'feedforward'
        return create_dl_model_feedforward(num_antennas, learning_rate, dropout_rate)


def prepare_dl_data(received_signal, transmitted_symbols, channel_vectors, desired_user_idx):
    """
    Prepares data for training/testing the DL model WITH steering vector.
    
    THIS IS THE KEY FIX: We include the steering vector so the model knows
    which direction to look for the desired user!
    
    Args:
        received_signal: (num_antennas x num_symbols) complex array
        transmitted_symbols: (num_users x num_symbols) complex array
        channel_vectors: (num_users x num_antennas) complex array - steering vectors
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
    # This gives the model BOTH the mixed signal AND which direction to look!
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
