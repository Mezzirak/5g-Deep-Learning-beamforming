import numpy as np
import matplotlib.pyplot as plt
from simulation import generate_simulation_data, generate_training_data
from models import mvdr_beamformer, create_dl_model, prepare_dl_data
from plotting import plot_constellation, plot_beam_pattern
from evaluate import calculate_ber, calculate_sinr
from tqdm import tqdm 
import tensorflow as tf


try:
    tf.config.set_visible_devices([], 'GPU')
    print("ℹ️  GPU disabled. Running on CPU", flush=True)
except:
    pass

# System parameters
NUM_ANTENNAS = 16  # Number of antennas at base station
NUM_USERS = 10     # Number of simultaneous users (1 desired + 9 interferers)

# Training parameters
TRAINING_SNR_RANGE = np.arange(-10, 11, 2)  # SNRs to train on: [-10, -8, ..., 10]
NUM_TRAINING_SYMBOLS_PER_SNR = 2000         
NUM_CHANNEL_REALISATIONS = 100              

# Testing parameters
TEST_SNR_RANGE = np.arange(-10, 12, 2)      # SNRs to test on
NUM_TEST_SYMBOLS = 1000000                  # Total symbols per SNR point

# Model parameters
EPOCHS = 100                                # Training epochs
BATCH_SIZE = 256                            
LEARNING_RATE = 0.0005                     
DROPOUT_RATE = 0.2                         

print("="*80)
print("5G BEAMFORMING: MVDR vs DEEP LEARNING COMPARISON")
print("="*80)
print(f"Configuration:")
print(f"  - Antennas: {NUM_ANTENNAS}")
print(f"  - Users: {NUM_USERS} (1 desired + {NUM_USERS-1} interferers)")
print(f"  - Training Data: {NUM_CHANNEL_REALISATIONS} realisations x {NUM_TRAINING_SYMBOLS_PER_SNR} symbols")
print(f"  - Batch Size: {BATCH_SIZE}")

#Generate the training data

print("Generating training data")

X_train, Y_train = generate_training_data(
    num_antennas=NUM_ANTENNAS,
    num_users=NUM_USERS,
    snr_range_db=TRAINING_SNR_RANGE,
    num_symbols_per_snr=NUM_TRAINING_SYMBOLS_PER_SNR,
    num_realisations=NUM_CHANNEL_REALISATIONS 
)

print(f"\nTotal training samples: {len(X_train):,}")
print(f"Input shape: {X_train.shape}")
print(f"Output shape: {Y_train.shape}")

#Train the DL model

print(" Training Deep Learning Model")

dl_model = create_dl_model(
    num_antennas=NUM_ANTENNAS,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
)

print("\nModel architecture:")
dl_model.summary()

print("\nTraining model")
history = dl_model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title('Model Training Progress')
plt.legend(); plt.grid(True); plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.title('Model Training Progress (MAE)')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig('images/training_history.png', dpi=150, bbox_inches='tight')
print("\n✓ Training complete! Saved history to images/training_history.png")

#Evaluate model

print("Evaluating on test data")

mvdr_ber_results = []
dl_ber_results = []
mvdr_sinr_results = []
dl_sinr_results = []

# Averaging Setup: Split the 1M symbols into 20 chunks of 50k
NUM_AVG_RUNS = 20
SYMBOLS_PER_RUN = NUM_TEST_SYMBOLS // NUM_AVG_RUNS 

for snr_db in tqdm(TEST_SNR_RANGE, desc="Testing SNR points", unit="SNR"):
    
    # Temporary lists for averaging
    temp_mvdr_ber = []
    temp_dl_ber = []
    temp_mvdr_sinr = []
    temp_dl_sinr = []
    
    # Run multiple small tests
    for _ in range(NUM_AVG_RUNS):
        sim_data = generate_simulation_data(
            NUM_ANTENNAS, NUM_USERS, snr_db, SYMBOLS_PER_RUN
        )
        
        received_signal = sim_data["received_signal"]
        transmitted_symbols = sim_data["transmitted_symbols"]
        channel_vectors = sim_data["channel_vectors"]
        desired_user_idx = sim_data["desired_user_idx"]
        noise_power = sim_data["noise_power"]
        original_desired_symbols = transmitted_symbols[desired_user_idx, :]
        
        # MVDR
        mvdr_est, mvdr_w = mvdr_beamformer(received_signal, channel_vectors, desired_user_idx)
        temp_mvdr_ber.append(calculate_ber(mvdr_est, original_desired_symbols))
        temp_mvdr_sinr.append(calculate_sinr(mvdr_w, channel_vectors, desired_user_idx, noise_power))
        
        # Deep Learning
        # FIXED: Now passing channel_vectors so model knows where to look!
        X_test, _ = prepare_dl_data(
            received_signal, transmitted_symbols, channel_vectors, desired_user_idx
        )
        pred = dl_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        dl_est = pred[:, 0] + 1j * pred[:, 1]
        
        temp_dl_ber.append(calculate_ber(dl_est, original_desired_symbols))
        
        # DL Effective SINR
        sig_p = np.mean(np.abs(dl_est)**2)
        err_p = np.mean(np.abs(dl_est - original_desired_symbols)**2)
        dl_sinr = 10 * np.log10(sig_p / err_p) if err_p > 0 else 100
        temp_dl_sinr.append(dl_sinr)

    # Average the results
    mvdr_ber = np.mean(temp_mvdr_ber)
    dl_ber = np.mean(temp_dl_ber)
    
    mvdr_ber_results.append(mvdr_ber)
    dl_ber_results.append(dl_ber)
    mvdr_sinr_results.append(np.mean(temp_mvdr_sinr))
    dl_sinr_results.append(np.mean(temp_dl_sinr))
    
    tqdm.write(f"  SNR={snr_db:3d} dB | MVDR BER: {mvdr_ber:.6f} | DL BER: {dl_ber:.6f}")

print("\n✓ Testing complete")

#save and plot results

print("Generating Plots")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: BER
ax1 = axes[0]
ax1.semilogy(TEST_SNR_RANGE, mvdr_ber_results, 'o-', linewidth=2, label='MVDR', color='blue')
ax1.semilogy(TEST_SNR_RANGE, dl_ber_results, 's-', linewidth=2, label='Deep Learning', color='red')
ax1.set_xlabel('SNR [dB]'); ax1.set_ylabel('BER'); ax1.set_title('BER Comparison')
ax1.grid(True, which="both", alpha=0.3); ax1.legend(); ax1.set_ylim(1e-6, 1)

# Plot 2: SINR
ax2 = axes[1]
ax2.plot(TEST_SNR_RANGE, mvdr_sinr_results, 'o-', linewidth=2, label='MVDR SINR', color='blue')
ax2.plot(TEST_SNR_RANGE, dl_sinr_results, 's-', linewidth=2, label='DL Effective SINR', color='red')
ax2.set_xlabel('SNR [dB]'); ax2.set_ylabel('SINR [dB]'); ax2.set_title('SINR Comparison')
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout()
plt.savefig('images/ber_vs_snr_comparison_final.png', dpi=150)
print("✓ Saved plot to images/ber_vs_snr_comparison_final.png")

np.savez('results.npz', snr=TEST_SNR_RANGE, mvdr=mvdr_ber_results, dl=dl_ber_results)
print("✓ Saved numerical results to results.npz")