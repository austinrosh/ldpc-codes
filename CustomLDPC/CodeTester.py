import numpy as np

def bsc_channel(bits, flip_prob):
    """ 
    Simulate a Binary Symmetric Channel (BSC).
    Each bit is flipped with probability flip_prob.
    """
    flips = np.random.rand(len(bits)) < flip_prob
    return np.bitwise_xor(bits, flips.astype(int))

def awgn_channel(bits, EbNo_dB):
    """
    Simulate an AWGN channel with BPSK modulation.
    Bits are mapped as: 0 -> +1, 1 -> -1.
    Noise variance is computpiped from the given Eb/No (in dB).
    """
    # BPSK modulation: 0 -> +1, 1 -> -1
    symbols = 1 - 2 * bits
    EbNo_linear = 10 ** (EbNo_dB / 10)
    # For BPSK with unit energy per bit, noise sigma is:
    sigma = np.sqrt(1 / (2 * EbNo_linear))
    noise = sigma * np.random.randn(len(symbols))
    received = symbols + noise
    # Hard decision demodulation: threshold at 0
    #demod_bits = (received < 0).astype(int)
    #return demod_bits
    return received

def simulate_channel(message, encoder, decoder, channel_type, channel_param):
    """
    Simulate transmission of a message over a specified channel.
    
    Parameters:
        message       : numpy array of bits (0s and 1s)
        encoder       : function to encode the message (returns encoded bits)
        decoder       : function to decode received bits (returns decoded message)
        channel_type  : 'BSC' or 'AWGN'
        channel_param : For BSC, flip probability; for AWGN, Eb/No in dB.
    """
    # Encode the message
    encoded = encoder(message)
    
    # Transmit over the selected channel
    if channel_type.upper() == 'BSC':
        received = bsc_channel(encoded, channel_param)
    elif channel_type.upper() == 'AWGN':
        received = awgn_channel(encoded, channel_param)
    else:
        raise ValueError("Unsupported channel type. Use 'BSC' or 'AWGN'.")
    
    # Decode the received bits
    decoded = decoder(received)
    return decoded

def simulate_trials(num_trials, message_length, encoder, decoder, channel_type, channel_param):
    """
    Run multiple simulation trials and compute the Bit Error Rate (BER).
    
    Parameters:
        num_trials    : number of independent trials to run.
        message_length: number of bits in each trial.
        encoder       : encoder function.
        decoder       : decoder function.
        channel_type  : channel type ('BSC' or 'AWGN').
        channel_param : channel parameter (flip probability for BSC or Eb/No in dB for AWGN).
        
    Returns:
        Bit Error Rate (BER) as a float.
    """
    total_bit_errors = 0
    total_bits = 0
    for _ in range(num_trials):
        message = np.random.randint(0, 2, message_length)
        decoded = simulate_channel(message, encoder, decoder, channel_type, channel_param)
        total_bit_errors += np.sum(message != decoded)
        total_bits += message_length
    return total_bit_errors / total_bits 

# ----- Example Encoder/Decoder Implementations -----

def identity_encoder(message):
    """No coding; returns the message as is."""
    return message

def identity_decoder(received):
    """No decoding; returns the received bits as is."""
    return received

def repetition_encoder(message, repeat=3):
    """
    Simple repetition code encoder.
    Each bit is repeated 'repeat' times.
    """
    return np.repeat(message, repeat)

def repetition_decoder(received, repeat=3):
    """
    Decoder for the repetition code.
    Uses majority voting on each group of repeated bits.
    """
    if len(received) % repeat != 0:
        raise ValueError("Length of received bits is not a multiple of repeat factor.")
    # Reshape to have one row per original bit
    received_reshaped = received.reshape(-1, repeat)
    # Majority vote: if sum > repeat/2, decide 1; otherwise 0.
    decoded = (np.sum(received_reshaped, axis=1) > (repeat / 2)).astype(int)
    return decoded

# ----- Main Simulation -----
if __name__ == '__main__':
    # Simulation parameters
    message_length = 1000     # bits per trial
    num_trials = 1         # number of trials to average over
    
    # Channel selection: choose 'BSC' or 'AWGN'
    channel_type = 'BSC'
    # For BSC, channel_param is the flip probability (e.g., 0.01)
    # For AWGN, channel_param is Eb/No in dB (e.g., 5.0)
    channel_param = .01      # Example: Eb/No = 5 dB for AWGN
    
    # Select encoder/decoder pair:
    # For uncoded (identity) transmission, use identity_encoder/identity_decoder.
    # To test a repetition code, uncomment the following two lines:
    # encoder = lambda m: repetition_encoder(m, repeat=3)
    # decoder = lambda r: repetition_decoder(r, repeat=3)
    
    # For this example, we use the uncoded system:
    encoder = repetition_encoder
    decoder = repetition_decoder
    
    # Run simulation
    ber = simulate_trials(num_trials, message_length, encoder, decoder, channel_type, channel_param)
    print(f"Channel Type: {channel_type}")
    print(f"Channel Parameter: {channel_param}")
    print(f"Bit Error Rate (BER): {ber*100:.4f}%")