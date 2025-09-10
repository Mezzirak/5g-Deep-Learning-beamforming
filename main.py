from simulation import generate_simulation_data
from models import mvdr_beamformer, create_dl_model, prepare_dl_data
from plotting import plot_constellation, plot_beam_pattern
from evaluate import calculate_ber, calculate_sinr 

#Simulation Setup
NUM_ANTENNAS = 16
NUM_USERS = 3
SNR_DB = 15
NUM_SYMBOLS = 5000

print("Generating simulation data")
sim_data = generate_simulation_data(NUM_ANTENNAS, NUM_USERS, SNR_DB, NUM_SYMBOLS)

# Unpack the data dictionary
received_signal = sim_data["received_signal"]
transmitted_symbols = sim_data["transmitted_symbols"]
channel_vectors = sim_data["channel_vectors"]
desired_user_idx = sim_data["desired_user_idx"]
noise_power = sim_data["noise_power"] # Unpack the new noise_power variable
original_desired_symbols = transmitted_symbols[desired_user_idx, :]

#MVDR Beamforming (Baseline)
print("Running MVDR beamformer")
mvdr_estimated_symbols, mvdr_weights = mvdr_beamformer(received_signal, channel_vectors, desired_user_idx)

#Deep Learning Beamforming
print("Preparing data for DL model")
X_train, Y_train = prepare_dl_data(received_signal, transmitted_symbols, desired_user_idx)

print("Creating and training DL model")
dl_model = create_dl_model(NUM_ANTENNAS)
dl_model.summary()
history = dl_model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

# Get predictions from the DL model
predicted_split = dl_model.predict(X_train)
dl_estimated_symbols = predicted_split[:, 0] + 1j * predicted_split[:, 1]

#Evaluation and Plotting
print("\nResults")

# Calculate metrics for MVDR
mvdr_ber = calculate_ber(mvdr_estimated_symbols, original_desired_symbols)
mvdr_sinr = calculate_sinr(mvdr_weights, channel_vectors, desired_user_idx, noise_power)

# Calculate metrics for DL
dl_ber = calculate_ber(dl_estimated_symbols, original_desired_symbols)
#Calculating the SINR for the DL model is more complex, so I am comparing on BER for now.

print(f"MVDR --> BER: {mvdr_ber:.4f} | SINR: {mvdr_sinr:.2f} dB")
print(f"Deep Learning --> BER: {dl_ber:.4f}")

# Plotting results
plot_constellation(mvdr_estimated_symbols, original_desired_symbols, "Constellation after MVDR")
plot_beam_pattern(mvdr_weights, sim_data["angles_of_arrival"], sim_data["antenna_indices"], "MVDR Beam Pattern")
plot_constellation(dl_estimated_symbols, original_desired_symbols, "Constellation after Deep Learning")