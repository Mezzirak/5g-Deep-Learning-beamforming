import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

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

#MVDR Beamforming Implementation

#Using the signal and channel data from the previous steps

#First, I am calculating the Covariance Matrix (R)
# the received_signal_with_noise has shape (NUM_ANTENNAS, NUM_SYMBOLS)
#Need to transpose it to (NUM_SYMBOLS, NUM_ANTENNAS) for the calculation
Y = received_signal_with_noise.T 
# The covariance matrix R is (Y^H * Y) / N_symbols
R = (Y.conj().T @ Y) / NUM_SYMBOLS

# 2. Calculate the MVDR Weights (w)
# Get the steering vector for the desired user
a_d = channel_vectors[desired_user_idx, :]

# Calculate the inverse of the covariance matrix
R_inv = np.linalg.inv(R)

# Numerator of the MVDR formula
numerator = R_inv @ a_d
# Denominator of the MVDR formula
denominator = a_d.conj().T @ R_inv @ a_d

# Final MVDR weights
w_mvdr = numerator / denominator

# 3. Apply the Beamforming Weights
# w_mvdr has shape (NUM_ANTENNAS,)
# received_signal_with_noise has shape (NUM_ANTENNAS, NUM_SYMBOLS)
# We need w_mvdr^H * y(t) for each symbol
estimated_symbols = w_mvdr.conj().T @ received_signal_with_noise

#Visualise the Result
# Plot the transmitted vs. estimated symbols (constellation diagram)
plt.figure(figsize=(8, 8))
plt.scatter(np.real(estimated_symbols), np.imag(estimated_symbols), alpha=0.5, label='Estimated Symbols')
# Plot the original transmitted symbols for the desired user
original_desired_symbols = transmitted_symbols[desired_user_idx, :]
plt.scatter(np.real(original_desired_symbols), np.imag(original_desired_symbols), c='red', marker='x', s=100, label='Original BPSK Symbols')
plt.title('Constellation Diagram after MVDR Beamforming')
plt.xlabel('In-Phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

#visualise the beam pattern
plt.figure()
angles = np.linspace(-np.pi/2, np.pi/2, 360)
steering_vectors_scan = np.exp(-1j * np.pi * np.sin(angles[:, np.newaxis]) * antenna_indices)
beam_pattern = np.abs(steering_vectors_scan @ w_mvdr.conj())**2

plt.plot(np.rad2deg(angles), 10 * np.log10(beam_pattern))
plt.title('MVDR Beam Pattern')
plt.xlabel('Angle (degrees)')
plt.ylabel('Gain (dB)')
plt.grid(True)
for angle in np.rad2deg(angles_of_arrival):
    plt.axvline(x=angle, color='r', linestyle='--')
plt.ylim(-50, 5)
plt.show()

#Preparing the data for the Deep Learning Model

# INPUT (X): The noisy received signal at the antennas
#Splitting the complex numbers into real and imaginary parts
# The shape of received_signal_with_noise is (NUM_ANTENNAS, NUM_SYMBOLS)
#Want the shape to be (NUM_SYMBOLS, NUM_ANTENNAS * 2)
X_complex = received_signal_with_noise.T # Shape (NUM_SYMBOLS, NUM_ANTENNAS)
X_train = np.hstack([np.real(X_complex), np.imag(X_complex)])

# OUTPUT (Y): The original, clean symbols for the desired user
Y_complex = transmitted_symbols[desired_user_idx, :].T # Shape (NUM_SYMBOLS)
Y_train = np.column_stack([np.real(Y_complex), np.imag(Y_complex)])

print(f"Shape of training input (X_train): {X_train.shape}")
print(f"Shape of training output (Y_train): {Y_train.shape}")

#Building and Training the Deep Learning Model

#Define the model architecture
model = Sequential([
    Input(shape=(NUM_ANTENNAS * 2,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2, activation='linear') # Output layer has 2 neurons for real and imag parts
])

#Compile the model
#Using the Mean Squared Error as the loss function because this is a regression problem
#The Adam optimiser is a standard and effective choice
model.compile(optimizer='adam', loss='mse')
model.summary()

#Train the model
print("\nTraining the Deep Learning model:")
# validation_split=0.2 means 20% of the data is set aside for validation
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

#Evaluate the DL Model
#Use the trained model to make predictions on the training data
predicted_symbols_dl_split = model.predict(X_train)
#Combine real and imaginary parts back into a complex number
predicted_symbols_dl = predicted_symbols_dl_split[:, 0] + 1j * predicted_symbols_dl_split[:, 1]

#Plot the constellation diagram for the DL model output
plt.figure(figsize=(8, 8))
plt.scatter(np.real(predicted_symbols_dl), np.imag(predicted_symbols_dl), alpha=0.5, label='DL Estimated Symbols')
plt.scatter(np.real(original_desired_symbols), np.imag(original_desired_symbols), c='red', marker='x', s=100, label='Original BPSK Symbols')
plt.title('Constellation Diagram after Deep Learning Beamforming')
plt.xlabel('In-Phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()