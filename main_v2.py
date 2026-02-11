"""
Enhanced 5G Beamforming: Low-SNR Improvement + Channel Estimation Errors

This script addresses two key limitations:

1. LOW-SNR PERFORMANCE IMPROVEMENT
   - Problem: Original DL underperforms MVDR at SNR < 0 dB
   - Solution: Generate 2× more training data at low SNR (weighted sampling)
   - Expected: Match or beat MVDR even at -10 dB

2. CHANNEL ESTIMATION ERRORS  
   - Problem: Original assumes perfect CSI (unrealistic)
   - Solution: Add Gaussian noise to channel estimates (σ² = 0.1)
   - Expected: Show DL is more robust to CSI errors than MVDR

Compares 5 configurations:
- MVDR with perfect CSI
- MVDR with imperfect CSI  
- Zero-Forcing with perfect CSI
- Zero-Forcing with imperfect CSI
- DL (Feedforward) with perfect CSI
- DL (Feedforward) with imperfect CSI (trained on noisy data)
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation_v2 import generate_simulation_data, generate_training_data
from models_v2 import (
    mvdr_beamformer, 
    zero_forcing_beamformer,
    create_dl_model_feedforward as create_dl_model,
    prepare_dl_data
)
import sys
sys.path.append('/mnt/user-data/uploads')
from evaluate import calculate_ber, calculate_sinr
from tqdm import tqdm
import tensorflow as tf

# Force CPU mode
try:
    tf.config.set_visible_devices([], 'GPU')
    print("ℹ️  GPU Disabled. Running on CPU for maximum speed.\n")
except:
    pass

#Configuration 

# System parameters
NUM_ANTENNAS = 16
NUM_USERS = 10

# Training parameters
TRAINING_SNR_RANGE = np.arange(-10, 11, 2)
NUM_TRAINING_SYMBOLS_PER_SNR = 2000
NUM_CHANNEL_REALISATIONS = 100

# Low-SNR Boost
LOW_SNR_BOOST = 2.0  # Generate 2× more samples for SNR < 0 dB

# Testing parameters
TEST_SNR_RANGE = np.arange(-10, 12, 2)
NUM_TEST_SYMBOLS = 1000000
NUM_AVG_RUNS = 20

# Model parameters
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.2

# Channel Estimation Error
CHANNEL_ERROR_VARIANCE = 0.1  # 0.0 = perfect, 0.1 = realistic, 0.3 = poor

print("enhanced version: low-SNR improvement + CSI errors")
print(f"Configuration:")
print(f"  - Antennas: {NUM_ANTENNAS}")
print(f"  - Users: {NUM_USERS} (1 desired + {NUM_USERS-1} interferers)")
print(f"  - Training Data: {NUM_CHANNEL_REALISATIONS} realisations")
print(f"  - Base symbols per SNR: {NUM_TRAINING_SYMBOLS_PER_SNR}")
print(f"")
print(f"Low-SNR Boost")
print(f"  - Multiplier: {LOW_SNR_BOOST}× for SNR < 0 dB")
print(f"  - Low-SNR samples: {int(NUM_TRAINING_SYMBOLS_PER_SNR * LOW_SNR_BOOST):,} per realisation")
print(f"  - High-SNR samples: {NUM_TRAINING_SYMBOLS_PER_SNR:,} per realisation")
print(f"")
print(f"Channel Estimation Errors")
print(f"  - Error variance (σ²): {CHANNEL_ERROR_VARIANCE}")
print(f"  - Equivalent SNR on pilots: ~{10*np.log10(1/CHANNEL_ERROR_VARIANCE):.1f} dB")
print(f"  - Will compare perfect vs imperfect CSI")

#Generate training data

print(" Generating Training Data")

# Training set 1: Perfect CSI with low-SNR boost
print("\nTraining set 1: perfect CSI + Low-SNR Boost")
X_train_perfect, Y_train_perfect = generate_training_data(
    num_antennas=NUM_ANTENNAS,
    num_users=NUM_USERS,
    snr_range_db=TRAINING_SNR_RANGE,
    num_symbols_per_snr=NUM_TRAINING_SYMBOLS_PER_SNR,
    num_realisations=NUM_CHANNEL_REALISATIONS,
    channel_error_variance=0.0,  # Perfect CSI
    low_snr_boost=LOW_SNR_BOOST  # MORE DATA AT LOW SNR
)

# Training set 2: Imperfect CSI with low-SNR boost
print("\nTraining set 2: imperfect CSI + Low-SNR boost")
X_train_imperfect, Y_train_imperfect = generate_training_data(
    num_antennas=NUM_ANTENNAS,
    num_users=NUM_USERS,
    snr_range_db=TRAINING_SNR_RANGE,
    num_symbols_per_snr=NUM_TRAINING_SYMBOLS_PER_SNR,
    num_realisations=NUM_CHANNEL_REALISATIONS,
    channel_error_variance=CHANNEL_ERROR_VARIANCE,  # Imperfect CSI
    low_snr_boost=LOW_SNR_BOOST  # MORE DATA AT LOW SNR
)

#train model

print("Training DL model")

# Model 1: DL trained on perfect CSI
print("\n Model 1: DL with perfect CSI")
model_perfect = create_dl_model(
    num_antennas=NUM_ANTENNAS,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
)
print("\nModel architecture:")
model_perfect.summary()

print("\nTraining on perfect CSI data (with low-SNR boost)...")
history_perfect = model_perfect.fit(
    X_train_perfect, Y_train_perfect,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# Model 2: DL trained on imperfect CSI
print("\n Model 2: DL with imperfect CSI")
model_imperfect = create_dl_model(
    num_antennas=NUM_ANTENNAS,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
)

print("\nTraining on imperfect CSI data (with low-SNR boost)")
print(f"(Training with noisy channels: σ² = {CHANNEL_ERROR_VARIANCE})")
history_imperfect = model_imperfect.fit(
    X_train_imperfect, Y_train_imperfect,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

print("\n✓ Both models trained!")

# Plot training histories side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Perfect CSI training
axes[0].plot(history_perfect.history['loss'], label='Training Loss')
axes[0].plot(history_perfect.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('DL Training: Perfect CSI + Low-SNR Boost')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Imperfect CSI training  
axes[1].plot(history_imperfect.history['loss'], label='Training Loss')
axes[1].plot(history_imperfect.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title(f'DL Training: Imperfect CSI (σ²={CHANNEL_ERROR_VARIANCE}) + Low-SNR Boost')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('images/training_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved training comparison to images/training_comparison.png")

#evaluation

print("\n" + "="*80)
print("PHASE 3: Evaluating on Test Data")
print("="*80)

# Storage for all results
results = {
    'mvdr_perfect': [],
    'mvdr_imperfect': [],
    'zf_perfect': [],
    'zf_imperfect': [],
    'dl_perfect': [],
    'dl_imperfect': []
}

SYMBOLS_PER_RUN = NUM_TEST_SYMBOLS // NUM_AVG_RUNS

for snr_db in tqdm(TEST_SNR_RANGE, desc="Testing SNR points", unit="SNR"):
    
    # Temporary storage for averaging
    temp = {key: [] for key in results.keys()}
    
    for _ in range(NUM_AVG_RUNS):
        
        # Generate test data with perfect CSI
        sim_perfect = generate_simulation_data(
            NUM_ANTENNAS, NUM_USERS, snr_db, SYMBOLS_PER_RUN,
            channel_error_variance=0.0
        )
        
        # Generate test data with imperfect CSI
        sim_imperfect = generate_simulation_data(
            NUM_ANTENNAS, NUM_USERS, snr_db, SYMBOLS_PER_RUN,
            channel_error_variance=CHANNEL_ERROR_VARIANCE
        )
        
        # Extract common data
        received_signal = sim_perfect["received_signal"]
        transmitted_symbols = sim_perfect["transmitted_symbols"]
        channel_perfect = sim_perfect["channel_vectors"]
        channel_imperfect = sim_imperfect["channel_vectors"]
        desired_idx = sim_perfect["desired_user_idx"]
        true_symbols = transmitted_symbols[desired_idx, :]
        
        #classical beamformers
        
        # MVDR - perfect CSI
        mvdr_out_p, _ = mvdr_beamformer(received_signal, channel_perfect, desired_idx)
        temp['mvdr_perfect'].append(calculate_ber(mvdr_out_p, true_symbols))
        
        # MVDR - imperfect CSI
        mvdr_out_i, _ = mvdr_beamformer(received_signal, channel_imperfect, desired_idx)
        temp['mvdr_imperfect'].append(calculate_ber(mvdr_out_i, true_symbols))
        
        # Zero-Forcing - perfect CSI
        zf_out_p, _ = zero_forcing_beamformer(received_signal, channel_perfect, desired_idx)
        temp['zf_perfect'].append(calculate_ber(zf_out_p, true_symbols))
        
        # Zero-Forcing - imperfect CSI
        zf_out_i, _ = zero_forcing_beamformer(received_signal, channel_imperfect, desired_idx)
        temp['zf_imperfect'].append(calculate_ber(zf_out_i, true_symbols))
        
        # DL trained on perfect CSI, tested on perfect CSI
        X_test_p, _ = prepare_dl_data(received_signal, transmitted_symbols, 
                                       channel_perfect, desired_idx)
        pred_p = model_perfect.predict(X_test_p, batch_size=BATCH_SIZE, verbose=0)
        dl_out_p = pred_p[:, 0] + 1j * pred_p[:, 1]
        temp['dl_perfect'].append(calculate_ber(dl_out_p, true_symbols))
        
        # DL trained on imperfect CSI, tested on imperfect CSI
        X_test_i, _ = prepare_dl_data(received_signal, transmitted_symbols, 
                                       channel_imperfect, desired_idx)
        pred_i = model_imperfect.predict(X_test_i, batch_size=BATCH_SIZE, verbose=0)
        dl_out_i = pred_i[:, 0] + 1j * pred_i[:, 1]
        temp['dl_imperfect'].append(calculate_ber(dl_out_i, true_symbols))
    
    # Average results over all runs
    for key in results.keys():
        results[key].append(np.mean(temp[key]))
    
    # Print progress
    tqdm.write(f"  SNR={snr_db:3d} dB | " +
               f"MVDR(perf)={results['mvdr_perfect'][-1]:.5f} | " +
               f"MVDR(imp)={results['mvdr_imperfect'][-1]:.5f} | " +
               f"DL(perf)={results['dl_perfect'][-1]:.5f} | " +
               f"DL(imp)={results['dl_imperfect'][-1]:.5f}")

print("\n✓ Testing complete!")

#results analysis and plotting

print("PHASE 4: Analysis & Plotting")

# Create  comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# perfect CSI
ax1 = axes[0, 0]
ax1.semilogy(TEST_SNR_RANGE, results['mvdr_perfect'], 'o-', linewidth=2.5, 
             label='MVDR (Perfect CSI)', colour='blue', markersize=7)
ax1.semilogy(TEST_SNR_RANGE, results['zf_perfect'], 's-', linewidth=2.5, 
             label='ZF (Perfect CSI)', colour='green', markersize=7)
ax1.semilogy(TEST_SNR_RANGE, results['dl_perfect'], 'D-', linewidth=2.5, 
             label='DL (Perfect CSI + Low-SNR Boost)', colour='red', markersize=7)

ax1.set_xlabel('SNR [dB]', fontsize=13)
ax1.set_ylabel('BER', fontsize=13)
ax1.set_title('Perfect CSI: Effect of Low-SNR Boost', fontsize=14, fontweight='bold')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=11, loc='upper right')
ax1.set_ylim(1e-6, 1)

# imperfect CSI
ax2 = axes[0, 1]
ax2.semilogy(TEST_SNR_RANGE, results['mvdr_imperfect'], 'o--', linewidth=2.5, 
             label=f'MVDR (σ²={CHANNEL_ERROR_VARIANCE})', colour='blue', 
             markersize=7, alpha=0.8)
ax2.semilogy(TEST_SNR_RANGE, results['zf_imperfect'], 's--', linewidth=2.5, 
             label=f'ZF (σ²={CHANNEL_ERROR_VARIANCE})', colour='green', 
             markersize=7, alpha=0.8)
ax2.semilogy(TEST_SNR_RANGE, results['dl_imperfect'], 'D--', linewidth=2.5, 
             label=f'DL (σ²={CHANNEL_ERROR_VARIANCE} + Low-SNR Boost)', colour='red', 
             markersize=7, alpha=0.8)

ax2.set_xlabel('SNR [dB]', fontsize=13)
ax2.set_ylabel('BER', fontsize=13)
ax2.set_title(f'Imperfect CSI: Robustness Test (σ²={CHANNEL_ERROR_VARIANCE})', 
              fontsize=14, fontweight='bold')
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=11, loc='upper right')
ax2.set_ylim(1e-6, 1)

# CSI degradation analysis
ax3 = axes[1, 0]

# Calculate degradation factor (how much worse with imperfect CSI)
mvdr_degradation = np.array(results['mvdr_imperfect']) / np.array(results['mvdr_perfect'])
zf_degradation = np.array(results['zf_imperfect']) / np.array(results['zf_perfect'])
dl_degradation = np.array(results['dl_imperfect']) / np.array(results['dl_perfect'])

ax3.plot(TEST_SNR_RANGE, mvdr_degradation, 'o-', linewidth=2.5, 
         label='MVDR Degradation', colour='blue', markersize=7)
ax3.plot(TEST_SNR_RANGE, zf_degradation, 's-', linewidth=2.5, 
         label='ZF Degradation', colour='green', markersize=7)
ax3.plot(TEST_SNR_RANGE, dl_degradation, 'D-', linewidth=2.5, 
         label='DL Degradation', colour='red', markersize=7)
ax3.axhline(y=1.0, colour='black', linestyle='--', alpha=0.5, label='No Degradation')

ax3.set_xlabel('SNR [dB]', fontsize=13)
ax3.set_ylabel('BER Ratio (Imperfect / Perfect)', fontsize=13)
ax3.set_title('CSI Error Impact: Robustness Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)
ax3.set_ylim(0.8, 3.0)

# low SNR performance
ax4 = axes[1, 1]

# Focus on low-SNR region
low_snr_mask = TEST_SNR_RANGE <= 2
snr_low = TEST_SNR_RANGE[low_snr_mask]

ax4.semilogy(snr_low, np.array(results['mvdr_perfect'])[low_snr_mask], 
             'o-', linewidth=3, label='MVDR (Perfect)', colour='blue', markersize=8)
ax4.semilogy(snr_low, np.array(results['dl_perfect'])[low_snr_mask], 
             'D-', linewidth=3, label='DL (Perfect + Boost)', colour='red', markersize=8)

ax4.set_xlabel('SNR [dB]', fontsize=13)
ax4.set_ylabel('BER', fontsize=13)
ax4.set_title(f'Low-SNR Focus: Effect of {LOW_SNR_BOOST}× Boost', 
              fontsize=14, fontweight='bold')
ax4.grid(True, which="both", alpha=0.3)
ax4.legend(fontsize=12)
ax4.set_xlim(snr_low[0]-0.5, snr_low[-1]+0.5)

plt.tight_layout()
plt.savefig('images/enhanced_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved analysis to images/enhanced_analysis.png")

# Save all numerical results
np.savez('results_enhanced.npz', 
         snr=TEST_SNR_RANGE,
         **results,
         channel_error_variance=CHANNEL_ERROR_VARIANCE,
         low_snr_boost=LOW_SNR_BOOST)
print("✓ Saved numerical results to results_enhanced.npz")

#summary stats

print(" Summary statistics")

# Find best SNR index for analysis (SNR = 10 dB)
best_snr_idx = np.where(TEST_SNR_RANGE == 10)[0][0]
low_snr_idx = np.where(TEST_SNR_RANGE == -10)[0][0]

print("\n Low-SNR Improvement")
print(f"At SNR = -10 dB (very noisy):")
print(f"  MVDR:  BER = {results['mvdr_perfect'][low_snr_idx]:.5f}")
print(f"  DL:    BER = {results['dl_perfect'][low_snr_idx]:.5f}")
improvement_low = (results['mvdr_perfect'][low_snr_idx] - results['dl_perfect'][low_snr_idx]) / results['mvdr_perfect'][low_snr_idx] * 100
if improvement_low > 0:
    print(f" DL improved by {improvement_low:.1f}% (thanks to low-SNR boost!)")
else:
    print(f"  DL still {-improvement_low:.1f}% worse (may need more boost or epochs)")

print("\n CSI Error Robustness")
print(f"At SNR = 10 dB with σ² = {CHANNEL_ERROR_VARIANCE}:")
print(f"\nMVDR:")
print(f"  Perfect CSI:   BER = {results['mvdr_perfect'][best_snr_idx]:.5f}")
print(f"  Imperfect CSI: BER = {results['mvdr_imperfect'][best_snr_idx]:.5f}")
mvdr_deg_pct = (results['mvdr_imperfect'][best_snr_idx] / results['mvdr_perfect'][best_snr_idx] - 1) * 100
print(f"  Degradation: +{mvdr_deg_pct:.1f}%")

print(f"\nDL:")
print(f"  Perfect CSI:   BER = {results['dl_perfect'][best_snr_idx]:.5f}")
print(f"  Imperfect CSI: BER = {results['dl_imperfect'][best_snr_idx]:.5f}")
dl_deg_pct = (results['dl_imperfect'][best_snr_idx] / results['dl_perfect'][best_snr_idx] - 1) * 100
print(f"  Degradation: +{dl_deg_pct:.1f}%")

print(f"\nRobustness Comparison:")
if dl_deg_pct < mvdr_deg_pct:
    robustness_factor = mvdr_deg_pct / dl_deg_pct
    print(f" DL is {robustness_factor:.1f}× MORE robust to CSI errors!")
else:
    print(f" MVDR is more robust (unexpected - check training)")

print("\n v2 EXPERIMENT COMPLETE!")
