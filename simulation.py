import numpy as np

def generate_simulation_data(num_antennas, num_users, snr_db, num_symbols):
    """
    Generates simulation data for a 5G multi-user environment.

    Returns:
        A dictionary containing all the necessary simulation data.
    """
    # Geometry Setup
    gNB_position = np.array([0, 0])
    user_radius_min, user_radius_max = 50, 200
    user_angles = np.random.uniform(-np.pi/2, np.pi/2, num_users)
    user_radii = np.random.uniform(user_radius_min, user_radius_max, num_users)
    user_positions = np.array([user_radii * np.cos(user_angles), user_radii * np.sin(user_angles)]).T

    desired_user_idx = 0
    interferer_indices = [i for i in range(num_users) if i != desired_user_idx]

    # Signal and Channel Generation
    transmitted_symbols = np.random.choice([-1, 1], size=(num_users, num_symbols))
    angles_of_arrival = np.arctan2(user_positions[:, 1], user_positions[:, 0])
    antenna_indices = np.arange(num_antennas)
    channel_vectors = np.exp(-1j * np.pi * np.sin(angles_of_arrival[:, np.newaxis]) * antenna_indices)

    # Received Signal and Noise
    received_signal_noiseless = channel_vectors.T @ transmitted_symbols
    signal_power = np.mean(np.abs(received_signal_noiseless)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*received_signal_noiseless.shape) + 1j * np.random.randn(*received_signal_noiseless.shape))
    received_signal_with_noise = received_signal_noiseless + noise

    # Package data into a dictionary for easy access
    sim_data = {
        "received_signal": received_signal_with_noise,
        "transmitted_symbols": transmitted_symbols,
        "channel_vectors": channel_vectors,
        "desired_user_idx": desired_user_idx,
        "angles_of_arrival": angles_of_arrival,
        "antenna_indices": antenna_indices
    }
    
    return sim_data