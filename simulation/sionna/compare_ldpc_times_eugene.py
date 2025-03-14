import os
import sys
sys.setrecursionlimit(5000)  # Increase recursion depth
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time  # for time measurements
import pandas as pd

import sionna
sionna.config.seed = 42
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.fec.conv import ConvEncoder, ViterbiDecoder, BCJRDecoder
from sionna.fec.turbo import TurboEncoder, TurboDecoder
from sionna.fec.linear import OSDecoder
from sionna.utils import BinarySource, ebnodb2no
from sionna.utils.metrics import count_block_errors
from sionna.channel import AWGN
from sionna.utils.plotting import PlotBER

###############################################################################
# Define the System Model
###############################################################################
class System_Model(tf.keras.Model):
    """System model for channel coding BER simulations.

    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to
    initialize the model.

    Parameters
    ----------
        k: int
            number of information bits per codeword.

        n: int
            codeword length.

        num_bits_per_symbol: int
            number of bits per QAM symbol.

        encoder: Keras layer
            A Keras layer that encodes information bit tensors.

        decoder: Keras layer
            A Keras layer that decodes llr tensors.

        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".

        sim_esno: bool
            A boolean; if True, `ebno_db` is interpreted as Es/N0 rather than Eb/N0.

        cw_estimates: bool
            If True, codewords instead of information estimates are returned.
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False,
                 cw_estimates=False):

        super().__init__()

        # Store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno
        self.cw_estimates = cw_estimates
        self.num_bits_per_symbol = num_bits_per_symbol

        # Initialize components
        self.source = BinarySource()

        # Constellation, Mapper, and Demapper
        self.constellation = Constellation("qam",
                                           num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)

        # The channel
        self.channel = AWGN()

        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder

    @tf.function  # enable graph mode for increased throughput
    def call(self, batch_size, ebno_db):

        # Calculate noise variance
        if self.sim_esno:
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=1,
                           coderate=1)
        else:
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)

        # Generate random data
        u = self.source([batch_size, self.k])
        c = self.encoder(u)       # encode
        x = self.mapper(c)        # map to symbols
        y = self.channel([x, no]) # transmit over AWGN
        llr_ch = self.demapper([y, no])  # demap to LLRs

        # Decode
        u_hat = self.decoder(llr_ch)

        if self.cw_estimates:
            return c, u_hat
        return u, u_hat

###############################################################################
# Throughput measurement function
###############################################################################
def get_throughput(batch_size, ebno_dbs, model, repetitions=3):
    """Simulate throughput in bit/s per ebno_dbs point."""
    throughput = np.zeros_like(ebno_dbs, dtype=float)

    # Dummy call to build the graph (if using @tf.function)
    _ = model(tf.constant(batch_size, tf.int32),
              tf.constant(0., tf.float32))

    for idx, ebno_db in enumerate(ebno_dbs):
        t_start = time.perf_counter()
        # Repeat the forward pass multiple times
        for _ in range(repetitions):
            u, u_hat = model(tf.constant(batch_size, tf.int32),
                             tf.constant(ebno_db, tf.float32))
        t_stop = time.perf_counter()
        # throughput in bit/s (size of u times #repetitions / total time)
        throughput[idx] = u.numpy().size * repetitions / (t_stop - t_start)

    return throughput

###############################################################################
# Decoding time measurement function (focuses on decode step only)
###############################################################################
def get_decode_time(batch_size, ebno_dbs, model, repetitions=50):
    """
    Measure the average decoding time (seconds) for each SNR in ebno_dbs.
    We do not time the entire forward pass, only the decode step.
    """
    decode_times = np.zeros_like(ebno_dbs, dtype=float)

    for idx, ebno_db in enumerate(ebno_dbs):
        single_runs = []
        for _ in range(repetitions):
            # 1) Generate data/encode/map
            u = model.source([batch_size, model.k])
            c = model.encoder(u)
            x = model.mapper(c)

            # 2) Noise variance
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=model.num_bits_per_symbol,
                           coderate=model.k/model.n)

            # 3) Pass through AWGN
            y = model.channel([x, no])

            # 4) Demap
            llr_ch = model.demapper([y, no])

            # 5) Time the decode only
            t0 = time.perf_counter()
            _ = model.decoder(llr_ch)  # decode
            t1 = time.perf_counter()

            single_runs.append(t1 - t0)

        decode_times[idx] = np.mean(single_runs)
    return decode_times

###############################################################################
# Main
###############################################################################
def main():
    # Block lengths we test
    ns = [1000, 2000, 4000]
    rate = 0.5  # fixed coderate

    codes_under_test = []
    for n in ns:
        k = int(rate * n)

        # 5G LDPC codes: Min-sum
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc, cn_type='minsum', num_iter=20)
        name = f"5G LDPC minsum, (n={n})"
        codes_under_test.append([enc, dec, name, k, n])

        # 5G LDPC codes: Sum-product
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc, cn_type='boxplus', num_iter=20)
        name = f"5G LDPC, sum-product, (n={n})"
        codes_under_test.append([enc, dec, name, k, n])

    # For plotting BER results
    ber_plot = PlotBER(f"BER Performance for Short/Long Code Lengths, R=1/2")

    # SNR range
    ebno_db = np.arange(-1, 5, 0.25)
    ebno_db_tput = [2.0]   # SNR to simulate throughput & decode times

    num_bits_per_symbol = 2  # QPSK
    num_bits_per_batch = 5e6
    num_repetitions = 20  # average throughput over multiple runs

    # Arrays to store throughput (just for demonstration)
    # We'll store results for each of the three block lengths in `ns`.
    n_codes = len(ns)
    throughput_ldpc_ms = np.zeros(n_codes)
    throughput_ldpc_spa = np.zeros(n_codes)

    # Arrays to store decode times
    decode_time_ms = np.zeros(n_codes)
    decode_time_spa = np.zeros(n_codes)

    # This will help map index of code to index in the arrays
    code_length = np.zeros(len(ns))

    # Run simulations
    for idx, code in enumerate(codes_under_test):
        print("\nRunning: " + code[2])
        batch_size = int(num_bits_per_batch / code[4])

        # Build system model
        model = System_Model(k=code[3],
                             n=code[4],
                             num_bits_per_symbol=num_bits_per_symbol,
                             encoder=code[0],
                             decoder=code[1])

        # BER simulation
        ber_plot.simulate(model,
                          ebno_dbs=ebno_db,
                          legend=code[2],
                          max_mc_iter=40,
                          num_target_block_errors=400,
                          batch_size=batch_size,
                          soft_estimates=False,
                          early_stop=True,
                          show_fig=True,
                          add_bler=False,
                          forward_keyboard_interrupt=True)

        # Map the code index to [0,1,2] for block lengths
        # idx in [0,2,4] -> min-sum; idx in [1,3,5] -> sum-product
        if idx in [0, 2, 4]:
            if idx == 0:
                idx_ = 0
            elif idx == 2:
                idx_ = 1
            else:
                idx_ = 2

            # Throughput
            thrpt = get_throughput(batch_size,
                                   ebno_db_tput,
                                   model,
                                   repetitions=num_repetitions)
            throughput_ldpc_ms[idx_] = thrpt[0]  # single SNR -> store first entry

            # Decode time (focus on decode step)
            dt = get_decode_time(batch_size,
                                 ebno_db_tput,
                                 model,
                                 repetitions=50)  # e.g., 50 repetitions
            decode_time_ms[idx_] = dt[0]

            code_length[idx_] = code[4]

        elif idx in [1, 3, 5]:
            if idx == 1:
                idx_ = 0
            elif idx == 3:
                idx_ = 1
            else:
                idx_ = 2

            thrpt = get_throughput(batch_size,
                                   ebno_db_tput,
                                   model,
                                   repetitions=num_repetitions)
            throughput_ldpc_spa[idx_] = thrpt[0]

            dt = get_decode_time(batch_size,
                                 ebno_db_tput,
                                 model,
                                 repetitions=50)
            decode_time_spa[idx_] = dt[0]

    # Finalize BER plot
    ber_plot(ylim=(1e-5, 1), show_bler=False, save_fig=True)

    print("Throughput (Min-sum) [bit/s]:", throughput_ldpc_ms)
    print("Throughput (Sum-product) [bit/s]:", throughput_ldpc_spa)
    print("Decode time (Min-sum) [s]:", decode_time_ms)
    print("Decode time (Sum-product) [s]:", decode_time_spa)

    ############################################################################
    # Plot throughput vs. N
    ############################################################################
    fig, ax = plt.subplots(figsize=(8,6))
    plt.grid(which="both")
    plt.title("Throughput @ Eb/N0 = 2.0 dB, R=1/2")
    plt.xlabel("Code Length, N")
    plt.ylabel("Throughput [Mbit/s]")
    x_tick_labels = code_length.astype(int)
    plt.plot(code_length, throughput_ldpc_ms/1e6, label="LDPC, min-sum", linewidth=2)
    plt.plot(code_length, throughput_ldpc_spa/1e6, label="LDPC, sum-product", linewidth=2)
    plt.legend()
    plt.show()

    ############################################################################
    # Save decode time data to a text file and also plot decode time vs. N
    ############################################################################
    decode_data = pd.DataFrame({
        "Blocklength": code_length.astype(int),
        "MinSumDecodeTime(s)": decode_time_ms,
        "SumProductDecodeTime(s)": decode_time_spa
    })

    # Save to text file (space- or comma-separated, your choice)
    decode_data.to_csv("decode_times.txt", index=False, sep=" ")
    print("\nSaved decode times to 'decode_times.txt'.")
    print(decode_data)

    # Plot decode time vs. block length
    plt.figure(figsize=(8,6))
    plt.grid(which="both")
    plt.title("Decoding Time @ Eb/N0 = 2.0 dB, R=1/2")
    plt.xlabel("Code Length, N")
    plt.ylabel("Avg Decoding Time [s]")
    plt.plot(code_length, decode_time_ms, label="Min-sum", linewidth=2)
    plt.plot(code_length, decode_time_spa, label="Sum-product", linewidth=2)
    plt.legend()
    plt.show()

###############################################################################
# Run main
###############################################################################
if __name__ == "__main__":
     main()