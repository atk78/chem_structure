from scipy.fftpack import ifftn


def f_in_terf(c_hat, kappa, K2):
    return kappa * ifftn(K2 * c_hat).real


def fbulk(c, W):
    return W * c**2 * (1 - c) * c**2


def dfdc(c, W):
    return 2 * W * (c * (1 - c)**2 - (1 - c) * c**2)
