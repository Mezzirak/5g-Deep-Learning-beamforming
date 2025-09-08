import numpy as np
import matplotlib.pyplot as plt

#Simulation Parameters
NUM_ANTENNAS = 16       # Number of antennas at the gNB (M)
NUM_USERS = 3           # Number of users in the cell (K)
SNR_DB = 15             # Signal-to-Noise Ratio in dB
NUM_SYMBOLS = 1000      # Number of data symbols to simulate

#Geometry Setup
gNB_position = np.array([0, 0])
user_radius_min, user_radius_max = 50, 200
user_angles = np.random.uniform(-np.pi/2, np.pi/2, NUM_USERS) # Users in front of array
user_radii = np.random.uniform(user_radius_min, user_radius_max, NUM_USERS)
user_positions = np.array([user_radii * np.cos(user_angles), user_radii * np.sin(user_angles)]).T

desired_user_idx = 0
interferer_indices = [i for i in range(NUM_USERS) if i != desired_user_idx]

#Signal and Channel Generation
#Generate random BPSK symbols (+1 or -1) for each user
transmitted_symbols = np.random.choice([-1, 1], size=(NUM_USERS, NUM_SYMBOLS))

#Calculate Angle of Arrival (AoA) for each user
angles_of_arrival = np.arctan2(user_positions[:, 1], user_positions[:, 0])

#Create the channel vectors (array steering vectors) for a ULA
antenna_indices = np.arange(NUM_ANTENNAS)
# Note: np.newaxis is used to enable broadcasting for element-wise multiplication
channel_vectors = np.exp(-1j * np.pi * np.sin(angles_of_arrival[:, np.newaxis]) * antenna_indices)

#Create the noiseless received signal at the gNB antennas
# H.T is the channel matrix (M x K), transmitted_symbols is (K x N_symbols)
received_signal_noiseless = channel_vectors.T @ transmitted_symbols

#Additive White Gaussian Noise (AWGN)
# Calculate signal power
signal_power = np.mean(np.abs(received_signal_noiseless)**2)

#Calculate the noise power based on the SNR
# SNR = 10 * log10(signal_power / noise_power)
snr_linear = 10**(SNR_DB / 10)
noise_power = signal_power / snr_linear

#Generate complex Gaussian noise
noise = np.sqrt(noise_power / 2) * (np.random.randn(*received_signal_noiseless.shape) + 1j * np.random.randn(*received_signal_noiseless.shape))

#Add noise to the signal
received_signal_with_noise = received_signal_noiseless + noise

print(f"Shape of received signal matrix: {received_signal_with_noise.shape}") # Should be (NUM_ANTENNAS, NUM_SYMBOLS)
print(f"Angle of arrival for desired user ({desired_user_idx}): {np.rad2deg(angles_of_arrival[desired_user_idx]):.2f} degrees")

#Visualisation
plt.figure(figsize=(8, 8))
plt.plot(gNB_position[0], gNB_position[1], 'rs', markersize=10, label='Base Station (gNB)')
plt.plot(user_positions[desired_user_idx, 0], user_positions[desired_user_idx, 1], 'bo', markersize=8, label=f'Desired User ({desired_user_idx})')
plt.plot(user_positions[interferer_indices, 0], user_positions[interferer_indices, 1], 'kx', markersize=8, label='Interfering Users')
for i in range(NUM_USERS):
    plt.plot([gNB_position[0], user_positions[i, 0]], [gNB_position[1], user_positions[i, 1]], 'k:')
plt.xlabel('X-coordinate (metres)')
plt.ylabel('Y-coordinate (metres)')
plt.title('5G Multi-User Simulation Environment')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()