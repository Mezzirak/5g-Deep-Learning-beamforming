import numpy as np
import matplotlib.pyplot as plt

def plot_constellation(estimated, original, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(estimated), np.imag(estimated), alpha=0.5, label='Estimated Symbols')
    plt.scatter(np.real(original), np.imag(original), c='red', marker='x', s=100, label='Original Symbols')
    plt.title(title)
    plt.xlabel('In-Phase (I)'); plt.ylabel('Quadrature (Q)')
    plt.grid(True); plt.legend(); plt.axis('equal'); plt.show()

def plot_beam_pattern(weights, angles_of_arrival, antenna_indices, title):
    plt.figure()
    angles_scan = np.linspace(-np.pi/2, np.pi/2, 360)
    steering_vectors_scan = np.exp(-1j * np.pi * np.sin(angles_scan[:, np.newaxis]) * antenna_indices)
    beam_pattern = np.abs(steering_vectors_scan @ weights.conj())**2
    
    plt.plot(np.rad2deg(angles_scan), 10 * np.log10(beam_pattern))
    plt.title(title)
    plt.xlabel('Angle (degrees)'); plt.ylabel('Gain (dB)')
    plt.grid(True); plt.ylim(-50, 5)
    for angle in np.rad2deg(angles_of_arrival):
        plt.axvline(x=angle, color='r', linestyle='--')
    plt.show()