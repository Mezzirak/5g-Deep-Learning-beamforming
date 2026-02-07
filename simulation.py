import numpy as np

def generate_simulation_data(num_antennas, num_users, snr_db, num_symbols):
    """
    Generates simulation data for a 5G multi-user environment
    
    Args:
        num_antennas: Number of antennas at base station
        num_users: Number of simultaneous users
        snr_db: Signal-to-Noise Ratio in decibels
        num_symbols: Number of symbols to generate
        
    Returns:
        Dictionary containing all simulation data
        
    Explanation of what each step does:
    1. Place users randomly in a semi-circle around the base station
    2. Generate random BPSK symbols (+1 or -1) for each user
    3. Create channel vectors based on angles of arrival (steering vectors)
    4. Mix all user signals together at the antenna array
    5. Add noise based on the specified SNR
    """
    # Geometry Setup - Place base station at origin
    gNB_position = np.array([0, 0])
    
    # Place users in a 180-degree arc (front of base station)
    # between 50m and 200m away
    user_radius_min, user_radius_max = 50, 200
    user_angles = np.random.uniform(-np.pi/2, np.pi/2, num_users)
    user_radii = np.random.uniform(user_radius_min, user_radius_max, num_users)
    user_positions = np.array([
        user_radii * np.cos(user_angles), 
        user_radii * np.sin(user_angles)
    ]).T

    # User 0 is our desired user, all others are interferers
    desired_user_idx = 0
    interferer_indices = [i for i in range(num_users) if i != desired_user_idx]

    # Signal Generation - BPSK modulation (binary phase shift keying)
    # Each symbol is either +1 or -1
    transmitted_symbols = np.random.choice([-1, 1], size=(num_users, num_symbols))
    
    # Channel Model - Create steering vectors for each user
    # These represent how signals from each direction arrive at the array
    angles_of_arrival = np.arctan2(user_positions[:, 1], user_positions[:, 0])
    antenna_indices = np.arange(num_antennas)
    
    # Steering vector formula: exp(-j*pi*sin(theta)*antenna_position)
    # This creates the spatial signature of each user
    channel_vectors = np.exp(
        -1j * np.pi * np.sin(angles_of_arrival[:, np.newaxis]) * antenna_indices
    )

    # Received Signal - Superposition of all user signals
    # Matrix multiplication: (num_antennas x num_users) @ (num_users x num_symbols)
    # Result: (num_antennas x num_symbols)
    received_signal_noiseless = channel_vectors.T @ transmitted_symbols
    
    # Noise Addition - Calculate noise power based on SNR
    # SNR = Signal_Power / Noise_Power
    signal_power = np.mean(np.abs(received_signal_noiseless)**2)
    snr_linear = 10**(snr_db / 10)  # Convert dB to linear scale
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise (divide by 2 for I and Q components)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*received_signal_noiseless.shape) + 
        1j * np.random.randn(*received_signal_noiseless.shape)
    )
    received_signal_with_noise = received_signal_noiseless + noise

    # Package everything into a dictionary for easy access
    sim_data = {
        "received_signal": received_signal_with_noise,
        "transmitted_symbols": transmitted_symbols,
        "channel_vectors": channel_vectors,
        "desired_user_idx": desired_user_idx,
        "angles_of_arrival": angles_of_arrival,
        "antenna_indices": antenna_indices,
        "noise_power": noise_power
    }
    
    return sim_data


def generate_training_data(num_antennas, num_users, snr_range_db, 
                          num_symbols_per_snr, num_realisations=10):
    """
    Generate comprehensive training dataset across multiple SNR levels.
    
    This creates a more robust training set by:
    1. Sampling from multiple SNR conditions
    2. Creating multiple channel realisations (different user positions)
    3. Aggregating into one large diverse dataset
    
    Args:
        num_antennas: Number of antennas
        num_users: Number of users per realisation
        snr_range_db: Array of SNR values to sample from (e.g., [-10, -5, 0, 5, 10])
        num_symbols_per_snr: Symbols to generate per SNR point
        num_realisations: Number of different channel realisations per SNR
        
    Returns:
        X_train: Input features (real and imag parts of received signal)
        Y_train: Target labels (real and imag parts of desired symbol)
    """
    X_list = []
    Y_list = []
    
    print(f"Generating training data across {len(snr_range_db)} SNR points...")
    
    for snr_db in snr_range_db:
        for realisation in range(num_realisations):
            # Generate a new channel realisation (different user positions)
            sim_data = generate_simulation_data(
                num_antennas, num_users, snr_db, num_symbols_per_snr
            )
            
            # Extract received signal and desired symbols
            received_signal = sim_data["received_signal"]
            desired_symbols = sim_data["transmitted_symbols"][
                sim_data["desired_user_idx"], :
            ]
            
            # Convert complex to real (split into I and Q components)
            X_complex = received_signal.T  # Shape: (num_symbols, num_antennas)
            X_real = np.hstack([np.real(X_complex), np.imag(X_complex)])
            
            Y_complex = desired_symbols
            Y_real = np.column_stack([np.real(Y_complex), np.imag(Y_complex)])
            
            X_list.append(X_real)
            Y_list.append(Y_real)
    
    # Concatenate all data and shuffle
    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    Y_train = Y_train[shuffle_idx]
    
    print(f"Training data generated: {len(X_train)} samples")
    
    return X_train, Y_train