import numpy as np
from tqdm import tqdm

def generate_simulation_data(num_antennas, num_users, snr_db, num_symbols):
    """
    Generates simulation data for a 5G multi-user environment.
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
    
    # Channel Model - Create steering vectors
    angles_of_arrival = np.arctan2(user_positions[:, 1], user_positions[:, 0])
    antenna_indices = np.arange(num_antennas)
    
    # Steering vector formula
    channel_vectors = np.exp(
        -1j * np.pi * np.sin(angles_of_arrival[:, np.newaxis]) * antenna_indices
    )

    # Received Signal - Superposition of all user signals
    received_signal_noiseless = channel_vectors.T @ transmitted_symbols
    
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
    """
    X_list = []
    Y_list = []
    
    print(f"Generating training data across {len(snr_range_db)} SNR points...")
    
    for snr_db in tqdm(snr_range_db, desc="Generating Data", unit="SNR"):
        for realisation in range(num_realisations):
            # Generate a new channel realisation
            sim_data = generate_simulation_data(
                num_antennas, num_users, snr_db, num_symbols_per_snr
            )
            
            # Extract data
            received_signal = sim_data["received_signal"]
            desired_symbols = sim_data["transmitted_symbols"][
                sim_data["desired_user_idx"], :
            ]
            
            # --- THE FIX IS HERE ---
            # Ensure we only take Real and Imag parts of the signal.
            # Dimensions: (Symbols x 16) -> (Symbols x 32)
            X_complex = received_signal.T 
            X_real = np.hstack([np.real(X_complex), np.imag(X_complex)])
            
            # Prepare targets
            Y_complex = desired_symbols
            Y_real = np.column_stack([np.real(Y_complex), np.imag(Y_complex)])
            
            X_list.append(X_real)
            Y_list.append(Y_real)
    
    # Concatenate all data
    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    Y_train = Y_train[shuffle_idx]
    
    print(f"Training data generated: {len(X_train)} samples")
    
    return X_train, Y_train