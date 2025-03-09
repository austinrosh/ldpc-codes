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
    
def plot_ber(snr_db,
             ber,
             legend="",
             ylabel="BER",
             title="Bit Error Rate",
             ebno=True,
             is_bler=None,
             xlim=None,
             ylim=None,
             save_fig=False,
             path=""):

    # legend must be a list or string
    if not isinstance(legend, list):
        assert isinstance(legend, str)
        legend = [legend]

    assert isinstance(title, str), "title must be str."

    # broadcast snr if ber is list
    if isinstance(ber, list):
        if not isinstance(snr_db, list):
            snr_db = [snr_db]*len(ber)

    # check that is_bler is list of same size and contains only bools
    if is_bler is None:
        if isinstance(ber, list):
            is_bler = [False] * len(ber) # init is_bler as list with False
        else:
            is_bler = False
    else:
        if isinstance(is_bler, list):
            assert (len(is_bler) == len(ber)), "is_bler has invalid size."
        else:
            assert isinstance(is_bler, bool), \
                "is_bler must be bool or list of bool."
            is_bler = [is_bler] # change to list

    # tile snr_db if not list, but ber is list

    fig, ax = plt.subplots(figsize=(16,10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(title, fontsize=18)
    # return figure handle
    if isinstance(ber, list):
        for idx, b in enumerate(ber):
            if is_bler[idx]:
                line_style = "--"
            else:
                line_style = ""
            plt.semilogy(snr_db[idx], b, line_style, linewidth=2)
    else:
        if is_bler:
            line_style = "--"
        else:
            line_style = ""
        plt.semilogy(snr_db, ber, line_style, linewidth=2)

    plt.grid(which="both")
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=18)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=18)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(legend, fontsize=14)
    
    if save_fig:
        plt.savefig(path)
        plt.close(fig)
    else:
        #plt.close(fig)
        pass
    return fig, ax

def main():
    ber_plot_ldpc = PlotBER(f"LDPC BER/BLER vs. Eb/N0, k=1000, Variable Rate")

    # Fixed information bits
    k = 1000  
    rates = [1/5, 1/3, 1/2, 5/6]  # Target rates
    ns = [round(k / rate) for rate in rates] 
    num_bits_per_symbol = [2, 2, 4, 6]  # QPSK
    mappings = ["QPSK","QPSK", "4-QAM", "64-QAM"]

    codes_under_test = []
    for i, n in enumerate(ns):
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc, cn_type='minsum', num_iter=20)
        name = f"R={rates[i]:.2f}, {mappings[i]}"
        codes_under_test.append([enc, dec, name, k, n])

    ebno_db_r1 = np.arange(0, 5, 0.25)  # Sim SNR range
    ebno_db_r2 = np.arange(0, 5, 0.25)  # Sim SNR range
    ebno_db_r3 = np.arange(3, 8, 0.25)  # Sim SNR range
    ebno_db_r4 = np.arange(8, 13, 0.25)  # Sim SNR range
    ebno_rates = [ebno_db_r1, ebno_db_r2, ebno_db_r3, ebno_db_r4]

    for idx, code in enumerate(codes_under_test):    
        print(f"Running: {code[2]}")
        model = System_Model(k=code[3], n=code[4], num_bits_per_symbol=num_bits_per_symbol[idx], encoder=code[0], decoder=code[1])

        # Simulate BER
        ber_plot_ldpc.simulate(model,
                               ebno_dbs=ebno_rates[idx],
                               legend=code[2],
                               max_mc_iter=100,
                               num_target_block_errors=500,
                               batch_size=1000,
                               soft_estimates=False,
                               early_stop=True,
                               add_bler=True,
                               show_fig=True,
                               forward_keyboard_interrupt=True)

    ber_plot_ldpc(ylim=(1e-5, 1), xlim=[0, 14])
    plt.show()

if __name__ == "__main__":
    main()