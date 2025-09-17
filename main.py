import numpy as np
import matplotlib.pyplot as plt
from simulation import generate_simulation_data
from models import mvdr_beamformer, create_dl_model, prepare_dl_data
from plotting import plot_constellation, plot_beam_pattern
from evaluate import calculate_ber, calculate_sinr

# Simulation Setup
NUM_ANTENNAS = 16
NUM_USERS = 3
NUM_SYMBOLS = 10000 # Use more symbols for more accurate BER

# Define the range of SNR values to test
snr_range_db = np.arange(0, 22, 2) # From 0dB to 20dB in steps of 2dB
mvdr_ber_results = []
dl_ber_results = []

# Main Experiment Loop
for snr_db in snr_range_db:
    print(f"\n--- Running simulation for SNR = {snr_db} dB")
    
    # Generate new data for each SNR point
    sim_data = generate_simulation_data(NUM_ANTENNAS, NUM_USERS, snr_db, NUM_SYMBOLS)
    
    # Unpack data
    received_signal = sim_data["received_signal"]
    transmitted_symbols = sim_data["transmitted_symbols"]
    channel_vectors = sim_data["channel_vectors"]
    desired_user_idx = sim_data["desired_user_idx"]
    original_desired_symbols = transmitted_symbols[desired_user_idx, :]

    # Run MVDR
    mvdr_estimated_symbols, _ = mvdr_beamformer(received_signal, channel_vectors, desired_user_idx)
    mvdr_ber = calculate_ber(mvdr_estimated_symbols, original_desired_symbols)
    mvdr_ber_results.append(mvdr_ber)
    
    # Prepare data and train/run DL model
    X_data, Y_data = prepare_dl_data(received_signal, transmitted_symbols, desired_user_idx)
    
    #train a new model for each noise level
    dl_model = create_dl_model(NUM_ANTENNAS)
    dl_model.fit(X_data, Y_data, epochs=50, batch_size=128, validation_split=0.2, verbose=0) # verbose=0 to keep logs clean
    
    predicted_split = dl_model.predict(X_data)
    dl_estimated_symbols = predicted_split[:, 0] + 1j * predicted_split[:, 1]
    dl_ber = calculate_ber(dl_estimated_symbols, original_desired_symbols)
    dl_ber_results.append(dl_ber)
    
    print(f"MVDR BER: {mvdr_ber:.5f} | DL BER: {dl_ber:.5f}")

# Plotting the final results
plt.figure(figsize=(10, 7))
plt.semilogy(snr_range_db, mvdr_ber_results, 'o-', label='MVDR Beamformer')
plt.semilogy(snr_range_db, dl_ber_results, 's-', label='Deep Learning Beamformer')
plt.xlabel('Signal-to-Noise Ratio (SNR) in dB')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Performance Comparison of Beamforming Algorithms')
plt.grid(True, which="both")
plt.legend()
plt.ylim(1e-5, 1)
plt.show()