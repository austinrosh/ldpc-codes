import sionna as sn
import numpy as np
import tensorflow as tf

# Ensure eager execution (needed for TensorFlow-based encoding/decoding)
tf.config.run_functions_eagerly(True)

# ✅ Choose valid 5G LDPC parameters (Base Graph 1 or 2)
n = 1024  # Codeword length
k = n/2
batch_size = 1  # Process one message at a time

# ✅ Correct way to initialize LDPC5GEncoder and LDPC5GDecoder
encoder = sn.fec.ldpc.encoding.LDPC5GEncoder(k=k, n=n)  # Only takes 'n'
decoder = sn.fec.ldpc.decoding.LDPC5GDecoder(encoder = encoder, num_iter = 20)  # Only 'n' and number of iterations

# ✅ Generate random message bits (shape must be [batch_size, k], and use the correct type)
message_bits = tf.random.uniform([batch_size, k], minval=0, maxval=2)

# ✅ Encode the message
codeword = encoder(message_bits)

# ✅ Decode the codeword
decoded_bits = decoder(codeword)

print("Original Message:", message_bits.numpy())
print("Decoded Message:", decoded_bits.numpy())

