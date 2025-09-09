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