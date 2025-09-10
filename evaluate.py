import numpy as np

def calculate_ber(estimated_symbols, original_symbols):
    """
    Calculates the Bit Error Rate (BER) for BPSK signals.
    """
    # Decode BPSK symbols by taking the sign of the real part
    decoded_bits = np.sign(np.real(estimated_symbols))
    original_bits = np.sign(np.real(original_symbols))
    
    # Count the number of errors
    num_errors = np.sum(decoded_bits != original_bits)
    total_bits = len(original_symbols)
    
    ber = num_errors / total_bits
    return ber

def calculate_sinr(weights, channel_vectors, desired_user_idx, noise_power):
    """
    Calculates the Signal-to-Interference-plus-Noise Ratio (SINR).
    """
    # Desired signal component
    h_d = channel_vectors[desired_user_idx, :]
    signal_power = np.abs(weights.conj().T @ h_d)**2
    
    # Interference component
    interference_power = 0
    for i in range(len(channel_vectors)):
        if i != desired_user_idx:
            h_i = channel_vectors[i, :]
            interference_power += np.abs(weights.conj().T @ h_i)**2
            
    # Noise component at the output
    # Noise power is amplified by the norm of the weight vector
    output_noise_power = noise_power * (np.linalg.norm(weights)**2)
    
    sinr = signal_power / (interference_power + output_noise_power)
    return 10 * np.log10(sinr) # Return SINR in dB