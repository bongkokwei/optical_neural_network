import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from functools import partial
from scipy.constants import c, pi


from sellmeier import sellmeier_mgln


def n_eff_tfln(wav):
    return sellmeier_mgln(wav, polarisation="e")


def wavevector(wav, n_eff):
    return 2 * pi * n_eff(wav * 1e6) / wav


def calculate_gvm(wave1, wave2, n_eff):
    k = partial(wavevector, n_eff=n_eff)

    dk1 = nd.Gradient(k, step=1e9)([wave1])
    dk2 = nd.Gradient(k, step=1e9)([wave2])

    return np.abs(dk1 - dk2)


if __name__ == "__main__":
    wav_array = np.linspace(1400e-9, 2010e-9, 100)
    gvm_array = [calculate_gvm(wav, wav / 2, n_eff=n_eff_tfln) for wav in wav_array]

    fig, ax = plt.subplots()
    ax.plot(wav_array * 1e9, gvm_array)
    ax.set_xlabel("wavelength (nm)")
    ax.grid(True)
    plt.show()
