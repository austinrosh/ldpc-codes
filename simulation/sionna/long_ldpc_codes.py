#Adapted from https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time # for throughput measurements


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

def get_throughput(batch_size, ebno_dbs, model, repetitions=1):
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

    # call model once to be sure it is compile properly
    # otherwise time to build graph is measured as well.
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
    ber_plot_ldpc = PlotBER(f"BER Performance of LDPC Codes @ Rate=1/2")


    # code parameters to simulate
    ns = [128, 256, 512, 1000, 2000]
          # 4000, 8000, 16000]  # number of codeword bits per codeword
    rate = 0.5 # fixed coderate

    # create list of encoder/decoder pairs to be analyzed
    codes_under_test = []
    # 5G LDPC codes
    for n in ns:
        k = int(rate*n) # calculate k for given n and rate
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc, cn_type='minsum',num_iter=20)
        name = f"5G LDPC minsum, (n={n})"
        codes_under_test.append([enc, dec, name, k, n])

    num_bits_per_symbol = 2 # QPSK

    ebno_db = np.arange(0, 3, 1) # sim SNR range
    # note that the waterfall for long codes can be steep and requires a fine
    # SNR quantization

    # run ber simulations for each case
    for code in codes_under_test:
        print("Running: " + code[2])
        model = System_Model(k=code[3],
                            n=code[4],
                            num_bits_per_symbol=num_bits_per_symbol,
                            encoder=code[0],
                            decoder=code[1])

        # the first argument must be a callable (function) that yields u and u_hat
        # for given batch_size and ebno
        # we fix the target number of BLOCK errors instead of the BER to
        # ensure that same accurate results for each block lengths is simulated
        ber_plot_ldpc.simulate(model, # the function have defined previously
                            ebno_dbs=ebno_db,
                            legend=code[2],
                            max_mc_iter=100,
                            num_target_block_errors=500, # we fix the target block errors
                            batch_size=1000,
                            soft_estimates=False,
                            early_stop=True,
                            show_fig=True,
                            forward_keyboard_interrupt=True); # should be True in a loop
    
        # throughput = get_throughput(batch_size=1000,
        #                     ebno_dbs=ebno_db, # snr point
        #                     model=model,
        #                     repetitions=20)

        # # print throughput
        # for idx, snr_db in enumerate(ebno_db):
        #     print(f"Throughput @ {snr_db:.1f} dB: {throughput[idx]/1e6:.2f} Mbit/s")

        # and show figure
    ber_plot_ldpc(ylim=(1e-5, 1), show_bler=False)
    plt.legend(fontsize=16)  # Set legend text size



    plt.show()

    # init binary source to generate information bits
    #source = BinarySource()
    # define a batch_size
   # batch_size = 1

    # generate random info bits
    #u = source([batch_size, k])

if __name__ == "__main__":
     main()
