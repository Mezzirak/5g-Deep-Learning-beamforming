import numpy as np
from tqdm import tqdm


def generate_simulation_data(num_antennas, num_users, snr_db, num_symbols, 
                            channel_error_variance=0.0):
    """
    Generates simulation data for a 5G multi-user environment.
    
    NEW: Added channel_error_variance parameter to simulate imperfect CSI
    
    Args:
        num_antennas: Number of antennas at base station
        num_users: Number of simultaneous users
        snr_db: Signal-to-Noise Ratio in decibels
        num_symbols: Number of symbols to generate
        channel_error_variance: CSI error variance (0 = perfect, 0.1 = realistic)
        
    Returns:
        Dictionary containing all simulation data
    """
    # Geometry Setup - Place base station at origin
    gNB_position = np.array([0, 0])
    
    # Place users in a 180-degree arc (front of base station)
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

    # Signal Generation - BPSK modulation (+1 or -1)
    transmitted_symbols = np.random.choice([-1, 1], size=(num_users, num_symbols))
    
    # Channel Model - Create PERFECT steering vectors
    angles_of_arrival = np.arctan2(user_positions[:, 1], user_positions[:, 0])
    antenna_indices = np.arange(num_antennas)
    
    # Perfect steering vectors
    channel_vectors_perfect = np.exp(
        -1j * np.pi * np.sin(angles_of_arrival[:, np.newaxis]) * antenna_indices
    )
    
    # Add channel estimation errors if specified
    if channel_error_variance > 0:
        # Add complex Gaussian noise to channel estimates
        noise_real = np.sqrt(channel_error_variance / 2) * np.random.randn(*channel_vectors_perfect.shape)
        noise_imag = np.sqrt(channel_error_variance / 2) * np.random.randn(*channel_vectors_perfect.shape)
        noise = noise_real + 1j * noise_imag
        
        channel_vectors_estimated = channel_vectors_perfect + noise
    else:
        channel_vectors_estimated = channel_vectors_perfect

    # Received Signal - Use PERFECT channels for signal generation
    # (Estimation error only affects beamformer, not the actual received signal)
    received_signal_noiseless = channel_vectors_perfect.T @ transmitted_symbols
    
    # Noise Addition
    signal_power = np.mean(np.abs(received_signal_noiseless)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*received_signal_noiseless.shape) + 
        1j * np.random.randn(*received_signal_noiseless.shape)
    )
    received_signal_with_noise = received_signal_noiseless + noise

    # Package everything into a dictionary
    sim_data = {
        "received_signal": received_signal_with_noise,
        "transmitted_symbols": transmitted_symbols,
        "channel_vectors": channel_vectors_estimated,  # ESTIMATED (noisy) channels
        "channel_vectors_perfect": channel_vectors_perfect,  # Store perfect for comparison
        "desired_user_idx": desired_user_idx,
        "angles_of_arrival": angles_of_arrival,
        "antenna_indices": antenna_indices,
        "noise_power": noise_power,
        "channel_error_variance": channel_error_variance
    }
    
    return sim_data


def generate_training_data(num_antennas, num_users, snr_range_db, 
                          num_symbols_per_snr, num_realisations=10,
                          channel_error_variance=0.0,
                          low_snr_boost=2.0):
    """
    Generate comprehensive training dataset across multiple SNR levels.
    
    NEW FEATURES:
    - channel_error_variance: Simulate imperfect CSI
    - low_snr_boost: Generate more samples at low SNR for better performance
    
    Args:
        num_antennas: Number of antennas
        num_users: Number of users per realisation
        snr_range_db: Array of SNR values to sample from
        num_symbols_per_snr: Base number of symbols per SNR point
        num_realisations: Number of different channel realisations per SNR
        channel_error_variance: CSI error variance (0 = perfect, 0.1 = realistic)
        low_snr_boost: Multiplier for samples at SNR < 0 dB (e.g., 2.0 = 2x more samples)
        
    Returns:
        X_train: Input features (received signal + steering vector)
        Y_train: Target labels (real and imag parts of desired symbol)
    """
    # Import here to avoid circular dependency
    from models_enhanced import prepare_dl_data
    
    X_list = []
    Y_list = []
    
    print(f"Generating training data across {len(snr_range_db)} SNR points...")
    if channel_error_variance > 0:
        print(f"⚠️  CSI Error Variance: {channel_error_variance:.3f} (imperfect channel estimation)")
    if low_snr_boost > 1.0:
        print(f"📊 Low-SNR boost: {low_snr_boost}x more samples for SNR < 0 dB")
    
    for snr_db in tqdm(snr_range_db, desc="Generating Data", unit="SNR"):
        # Apply low-SNR boost: more samples for difficult conditions
        if snr_db < 0:
            num_symbols_this_snr = int(num_symbols_per_snr * low_snr_boost)
        else:
            num_symbols_this_snr = num_symbols_per_snr
            
        for realisation in range(num_realisations):
            # Generate a new channel realisation
            sim_data = generate_simulation_data(
                num_antennas, num_users, snr_db, num_symbols_this_snr,
                channel_error_variance=channel_error_variance
            )
            
            # Use the FIXED prepare_dl_data that includes steering vectors
            X_real, Y_real = prepare_dl_data(
                received_signal=sim_data["received_signal"],
                transmitted_symbols=sim_data["transmitted_symbols"],
                channel_vectors=sim_data["channel_vectors"],  # Use ESTIMATED channels
                desired_user_idx=sim_data["desired_user_idx"]
            )
            
            X_list.append(X_real)
            Y_list.append(Y_real)
    
    # Concatenate all data
    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    Y_train = Y_train[shuffle_idx]
    
    print(f"Training data generated: {len(X_train):,} samples")
    print(f"✓ Input shape: {X_train.shape} (should be (N, 64) with steering vectors)")
    
    return X_train, Y_train
