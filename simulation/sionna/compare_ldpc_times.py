# Adapted from https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html
import os
import sys
sys.setrecursionlimit(5000)  # Increase recursion depth
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time # for throughput measurements
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
from sionna.utils.metrics import  count_block_errors
from sionna.channel import AWGN
from sionna.utils.plotting import PlotBER

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
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.

         cw_estiamtes: bool
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (u, u_hat):
            Tuple:

        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
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

        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned

        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol

        # init components
        self.source = BinarySource()

        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)

        # the channel can be replaced by more sophisticated models
        self.channel = AWGN()

        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder

    @tf.function() # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):

        # calculate noise variance
        if self.sim_esno:
                no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else:
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)

        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u) # explicitly encode

        x = self.mapper(c) # map c to symbols x

        y = self.channel([x, no]) # transmit over AWGN channel

        llr_ch = self.demapper([y, no]) # demap y to LLRs

        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)

        if self.cw_estimates:
            return c, u_hat

        return u, u_hat
    
def get_throughput(batch_size, ebno_dbs, model, repetitions=3):
    """ Simulate throughput in bit/s per ebno_dbs point.

    The results are average over `repetition` trials.

    Input
    -----
    batch_size: tf.int32
        Batch-size for evaluation.

    ebno_dbs: tf.float32
        A tensor containing SNR points to be evaluated.

    model:
        Function or model that yields the transmitted bits `u` and the
        receiver's estimate `u_hat` for a given ``batch_size`` and
        ``ebno_db``.

    repetitions: int
        An integer defining how many trails of the throughput
        simulation are averaged.

    """
    throughput = np.zeros_like(ebno_dbs)


    u, u_hat = model(tf.constant(batch_size, tf.int32),
                     tf.constant(0., tf.float32))

    for idx, ebno_db in enumerate(ebno_dbs):

        t_start = time.perf_counter()
        # average over multiple runs
        for _ in range(repetitions):
            u, u_hat = model(tf.constant(batch_size, tf.int32),
                             tf.constant(ebno_db, tf. float32))
        t_stop = time.perf_counter()
        # throughput in bit/s
        throughput[idx] = np.size(u.numpy())*repetitions / (t_stop - t_start)

    return throughput


def main():

    ns = [1000, 2000, 4000]
    rate = 0.5 # fixed coderate

    codes_under_test = []
    for n in ns:
        k = int(rate*n) # calculate k for given n and rate
        # 5G LDPC codes with 20 BP iterations
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc,cn_type='minsum',num_iter=20)
        name = f"5G LDPC minsum, (n={n})"
        codes_under_test.append([enc, dec, name, k, n])
        

        # Conv. code with Viterbi decoding
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc,cn_type='boxplus',num_iter=20)
        name = f"5G LDPC, sum-product, (n={n})"
        codes_under_test.append([enc, dec, name, k, n])



    
        

    ber_plot = PlotBER(f"BER Performance for Short/Long Code Lengths, R=1/2")

    num_bits_per_symbol = 2 # QPSK
    ebno_db = np.arange(-1, 5, 0.25) # sim SNR range
    ebno_db_tput = [2] # SNR to simulate
    num_bits_per_batch = 5e6 # must be reduced in case of out-of-memory errors
    num_repetitions = 20 # average throughput over multiple runs

    # run throughput simulations for each code
    n_codes = len(ns)

    throughput_ldpc_ms = np.zeros(n_codes)
    throughput_ldpc_spa = np.zeros(n_codes)
 
    code_length = np.zeros(len(ns))

    # run ber simulations for each code we have added to the list
    for idx, code in enumerate(codes_under_test): 
        print("\nRunning: " + code[2])
        batch_size = int(num_bits_per_batch / code[4])
        # generate a new model with the given encoder/decoder
        model = System_Model(k=code[3],
                            n=code[4],
                            num_bits_per_symbol=num_bits_per_symbol,
                            encoder=code[0],
                            decoder=code[1])

        # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
        ber_plot.simulate(model, # the function have defined previously
                            ebno_dbs=ebno_db, 
                            legend=code[2], 
                            max_mc_iter=40, 
                            num_target_block_errors=400, 
                            batch_size=batch_size, 
                            soft_estimates=False, 
                            early_stop=True,
                            show_fig=True, 
                            add_bler=False,
                            forward_keyboard_interrupt=True); 

       
        idx_ = 0
        if idx in [0, 2, 4]:
            if idx == 0:
                idx_ = 0
            elif idx == 2:
                idx_ = 1
            elif idx == 4:
                idx_ = 2

            throughput_ldpc_ms[idx_] = get_throughput(batch_size,
                                        ebno_db_tput,
                                        model,
                                        repetitions=num_repetitions)
            code_length[idx_] = code[4]

        elif idx in [1, 3, 5]:
            if idx == 1:
                idx_ = 0
            elif idx == 3:
                idx_ = 1
            elif idx == 5:
                idx_ = 2
            
            throughput_ldpc_spa[idx_] = get_throughput(batch_size,
                                        ebno_db_tput,
                                        model,
                                        repetitions=num_repetitions)


    ber_plot(ylim=(1e-5, 1), show_bler=False, save_fig=True) 
    print(throughput_ldpc_ms)
 


    # plots_to_show = ['5G LDPC minsum, (n=1000)', 'Conv. Code w/ Viterbi, (n=1000)', 'Turbo Code, (n=1000)', 
    #                  '5G LDPC minsum, (n=2000)', 'Conv. Code w/ Viterbi, (n=2000)', 'Turbo Code, (n=2000)', 
    #             '5G LDPC minsum, (n=4000)', 'Conv. Code w/ Viterbi, (n=4000)', 'Turbo Code, (n=4000)',
    #             '5G LDPC minsum, (n=8000)', 'Conv. Code w/ Viterbi, (n=8000)', 'Turbo Code, (n=8000)']
    
#     plots_to_show = ['5G LDPC minsum, (n=1000)', 'Conv. Code w/ Viterbi, (n=1000)', 'Turbo Code, (n=1000)', 
#                 '5G LDPC minsum, (n=8000)', 'Conv. Code w/ Viterbi, (n=8000)', 'Turbo Code, (n=8000)']


# #                '5G LDPC minsum, (n=8000)', 'Conv. Code w/ Viterbi, (n=8000)', 'Turbo Code, (n=8000)']
#     idx = []
#     for p in plots_to_show:
#         for i,l in enumerate(ber_plot._legends):
#             if p==l:
#                 idx.append(i)

#     current_directory = os.getcwd()
#     ber_data = {"Eb/N0 (dB)": ebno_db}

#     for i in idx:
#         legend_name = ber_plot._legends[i]
#         ber_values = ber_plot._bers[i]
#         ber_data[legend_name] = ber_values

#     df = pd.DataFrame(ber_data)
#     csv_filename = os.path.join(current_directory, "ber_results.csv")
#     df.to_csv(csv_filename, index=False)
#     print(f"BER data saved to {csv_filename}")


#     fig, ax = plt.subplots(figsize=(16,12))
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title(f"BER Performance for Increasing Code Size ", fontsize=25)
#     plt.grid(which="both")
#     plt.xlabel("$Eb/N0$ (dB)", fontsize=25)
#     plt.ylabel("BER", fontsize=25)

    # for i in range(int(len(idx)/4)):
    #     plt.semilogy(ebno_db,
    #                 ber_plot._bers[i],
    #                 c='C%d'%(i),
    #                 label=ber_plot._legends[idx[i]],
    #                 linewidth=2)
    #     plt.semilogy(ebno_db,
    #                 ber_plot._bers[idx[i+3]],
    #                 c='C%d'%(i),
    #                 label= ber_plot._legends[idx[i+3]],
    #                 linestyle = ":",
    #                 linewidth=2)
    #     plt.semilogy(ebno_db,
    #                 ber_plot._bers[idx[i+3+3]],
    #                 c='C%d'%(i),
    #                 label= ber_plot._legends[idx[i+3+3]],
    #                 linestyle = "--",
    #                 linewidth=2)
    #     plt.semilogy(ebno_db,
    #                 ber_plot._bers[idx[i+3+3+3]],
    #                 c='C%d'%(i),
    #                 label= ber_plot._legends[idx[i+3+3+3]],
    #                 linestyle = "-.",
    #                 linewidth=2)
        
    # for i in range(int(len(idx)/2)):
    #     plt.semilogy(ebno_db,
    #                 ber_plot._bers[i],
    #                 c='C%d'%(i),
    #                 label=ber_plot._legends[idx[i]],
    #                 linewidth=2)
    #     plt.semilogy(ebno_db,
    #                 ber_plot._bers[idx[i+3]],
    #                 c='C%d'%(i),
    #                 label= ber_plot._legends[idx[i+3]],
    #                 linestyle = "-.",
    #                 linewidth=2)


    # plt.legend(fontsize=12)
    # #plt.xlim([0, 4.5])
    # plt.ylim([1e-4, 1])

    fig, ax = plt.subplots(figsize=(16,12))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which="both")
    plt.title(f"Throughput, Eb/N0 = 3.0 dB, R=1/2", fontsize=18)
    plt.xlabel("Code Length, N", fontsize=16)
    plt.ylabel("Throughput, [Mbit/s]", fontsize=16)
    x_tick_labels = code_length.astype(int)
    plt.xticks(ticks=np.log2(code_length),labels=x_tick_labels, fontsize=16)
    plt.plot(np.log2(code_length), throughput_ldpc_ms/1e6, label="LDPC, min-sum", linewidth=2 )
    #plt.plot(np.log2(code_length), throughput_polar/1e6,label="Polar", linewidth=2)
    plt.plot(np.log2(code_length), throughput_ldpc_spa/1e6, label="LDPC, sum-product", linewidth=2)
    plt.legend(fontsize=12)
    plt.show()
    




if __name__ == "__main__":
     main()
