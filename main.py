import numpy as np
import matplotlib.pyplot as plt
from simulation import generate_simulation_data, generate_training_data
from models import mvdr_beamformer, create_dl_model, prepare_dl_data
from plotting import plot_constellation, plot_beam_pattern
from evaluate import calculate_ber, calculate_sinr
from tqdm import tqdm  # Progress bars!

#simulation configuration

# System parameters
NUM_ANTENNAS = 16  # Number of antennas at base station
NUM_USERS = 10     # Number of simultaneous users (1 desired + 9 interferers)

# Training parameters
TRAINING_SNR_RANGE = np.arange(-10, 11, 2)  # SNRs to train on: [-10, -8, ..., 10]
NUM_TRAINING_SYMBOLS_PER_SNR = 50000        # Symbols per SNR during training
NUM_CHANNEL_REALIZATIONS = 5                # Different channel configs per SNR

# Testing parameters
TEST_SNR_RANGE = np.arange(-10, 12, 2)      # SNRs to test on: [-10, -8, ..., 10]
NUM_TEST_SYMBOLS = 1000000                  # 1M symbols for statistical validity
                                            # Allows measuring BER down to ~10^-5

# Model parameters
EPOCHS = 100                               # Training epochs for DL model
BATCH_SIZE = 256                           # Batch size for training
LEARNING_RATE = 0.0005                     # Learning rate for Adam optimiser
DROPOUT_RATE = 0.2                         # Dropout rate for regularisation

print("="*80)
print("5G BEAMFORMING: MVDR vs DEEP LEARNING COMPARISON")
print("="*80)
print(f"\nConfiguration:")
print(f"  - Antennas: {NUM_ANTENNAS}")
print(f"  - Users: {NUM_USERS} (1 desired + {NUM_USERS-1} interferers)")
print(f"  - Training symbols per SNR: {NUM_TRAINING_SYMBOLS_PER_SNR:,}")
print(f"  - Testing symbols per SNR: {NUM_TEST_SYMBOLS:,}")
print(f"  - Channel realizations: {NUM_CHANNEL_REALIZATIONS}")
print(f"  - Training SNR range: {TRAINING_SNR_RANGE[0]} to {TRAINING_SNR_RANGE[-1]} dB")
print(f"  - Testing SNR range: {TEST_SNR_RANGE[0]} to {TEST_SNR_RANGE[-1]} dB")

#generate training data

print("\n" + "="*80)
print("PHASE 1: Generating Training Data")
print("="*80)

X_train, Y_train = generate_training_data(
    num_antennas=NUM_ANTENNAS,
    num_users=NUM_USERS,
    snr_range_db=TRAINING_SNR_RANGE,
    num_symbols_per_snr=NUM_TRAINING_SYMBOLS_PER_SNR,
    num_realizations=NUM_CHANNEL_REALIZATIONS
)

print(f"\nTotal training samples: {len(X_train):,}")
print(f"Input shape: {X_train.shape}")
print(f"Output shape: {Y_train.shape}")

#train deep learning model

print("\n" + "="*80)
print("PHASE 2: Training Deep Learning Model")
print("="*80)

# Create and train the model ONCE on all training data
dl_model = create_dl_model(
    num_antennas=NUM_ANTENNAS,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
)

print("\nModel Architecture:")
dl_model.summary()

print("\nTraining model")
print("(validation loss should follow training loss)")
history = dl_model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,  # Use 20% of data for validation
    verbose=1  # Shows progress bar per epoch
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Model Training Progress')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Model Training Progress (MAE)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/training_history.png', dpi=150, bbox_inches='tight')
print("\n✓ Training complete! Saved training history to images/training_history.png")

#evaluate on test data

print("\n" + "="*80)
print("PHASE 3: Evaluating on test data")
print("="*80)
print("This will take ~10-15 minutes (processing 11 million symbols total)")

mvdr_ber_results = []
dl_ber_results = []
mvdr_sinr_results = []
dl_sinr_results = []

# Main experiment loop with progress bar
for snr_db in tqdm(TEST_SNR_RANGE, desc="Testing SNR points", unit="SNR"):
    
    # Generate test data
    # Using 1M symbols for statistically valid BER measurements
    sim_data = generate_simulation_data(
        NUM_ANTENNAS, NUM_USERS, snr_db, NUM_TEST_SYMBOLS
    )
    
    # Unpack simulation data
    received_signal = sim_data["received_signal"]
    transmitted_symbols = sim_data["transmitted_symbols"]
    channel_vectors = sim_data["channel_vectors"]
    desired_user_idx = sim_data["desired_user_idx"]
    noise_power = sim_data["noise_power"]
    original_desired_symbols = transmitted_symbols[desired_user_idx, :]
    
    # Test the MVDR Beamformer
    mvdr_estimated_symbols, mvdr_weights = mvdr_beamformer(
        received_signal, channel_vectors, desired_user_idx
    )
    mvdr_ber = calculate_ber(mvdr_estimated_symbols, original_desired_symbols)
    mvdr_sinr = calculate_sinr(
        mvdr_weights, channel_vectors, desired_user_idx, noise_power
    )
    mvdr_ber_results.append(mvdr_ber)
    mvdr_sinr_results.append(mvdr_sinr)
    
    # Test Deep Learning Model
    X_test, Y_test = prepare_dl_data(
        received_signal, transmitted_symbols, desired_user_idx
    )
    
    # Predict using the trained model
    predicted_split = dl_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    dl_estimated_symbols = predicted_split[:, 0] + 1j * predicted_split[:, 1]
    
    dl_ber = calculate_ber(dl_estimated_symbols, original_desired_symbols)
    dl_ber_results.append(dl_ber)
    
    # For DL, compute effective SINR from symbol estimates
    dl_signal_power = np.mean(np.abs(dl_estimated_symbols)**2)
    dl_error_power = np.mean(np.abs(dl_estimated_symbols - original_desired_symbols)**2)
    dl_sinr = 10 * np.log10(dl_signal_power / dl_error_power) if dl_error_power > 0 else 100
    dl_sinr_results.append(dl_sinr)
    
    # Update progress bar description with current results
    tqdm.write(f"  SNR={snr_db:3d} dB | MVDR BER: {mvdr_ber:.6f} | DL BER: {dl_ber:.6f} | Improvement: {mvdr_ber/dl_ber if dl_ber > 0 else float('inf'):.2f}x")

print("\n✓ Testing complete!")

#plot results

print("\n" + "="*80)
print("PHASE 4: Generating Plots")
print("="*80)

# Creating comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: BER vs SNR
ax1 = axes[0]
ax1.semilogy(TEST_SNR_RANGE, mvdr_ber_results, 'o-', linewidth=2, 
             markersize=8, label='MVDR Beamformer', color='blue')
ax1.semilogy(TEST_SNR_RANGE, dl_ber_results, 's-', linewidth=2, 
             markersize=8, label='Deep Learning Beamformer', color='red')
ax1.set_xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12)
ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
ax1.set_title('BER Performance Comparison', fontsize=14, fontweight='bold')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim(1e-6, 1)

# Plot 2: SINR vs SNR
ax2 = axes[1]
ax2.plot(TEST_SNR_RANGE, mvdr_sinr_results, 'o-', linewidth=2, 
         markersize=8, label='MVDR SINR', color='blue')
ax2.plot(TEST_SNR_RANGE, dl_sinr_results, 's-', linewidth=2, 
         markersize=8, label='DL Effective SINR', color='red')
ax2.set_xlabel('Input SNR [dB]', fontsize=12)
ax2.set_ylabel('Output SINR [dB]', fontsize=12)
ax2.set_title('SINR Performance Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('images/ber_vs_snr_comparison_fixed.png', dpi=150, bbox_inches='tight')
print("✓ Saved comparison plot to images/ber_vs_snr_comparison_fixed.png")

# Save results to file
print("\n✓ Saving numerical results to results.npz...")
np.savez('results.npz',
         snr_range=TEST_SNR_RANGE,
         mvdr_ber=mvdr_ber_results,
         dl_ber=dl_ber_results,
         mvdr_sinr=mvdr_sinr_results,
         dl_sinr=dl_sinr_results)

# summary

print("\n" + "="*80)
print("EXPERIMENT COMPLETE - SUMMARY")
print("="*80)

print("\nFinal BER Results:")
print(f"{'SNR [dB]':<10} {'MVDR BER':<15} {'DL BER':<15} {'Improvement':<15}")
print("-" * 60)
for i, snr in enumerate(TEST_SNR_RANGE):
    improvement = mvdr_ber_results[i] / dl_ber_results[i] if dl_ber_results[i] > 0 else float('inf')
    print(f"{snr:<10} {mvdr_ber_results[i]:<15.6f} {dl_ber_results[i]:<15.6f} {improvement:<15.2f}x")

print("\n" + "="*80)
print("All results saved to:")
print("  - images/ber_vs_snr_comparison_fixed.png")
print("  - images/training_history.png")
print("  - results.npz")
print("="*80)
print("\n🎉 all done")