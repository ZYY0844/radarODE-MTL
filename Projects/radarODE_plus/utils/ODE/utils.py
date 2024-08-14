import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def index_convert(index, freq_org=30, freq_desired=200):
    """
    :param index: the index of the signal with freq=freq_org
    :param freq_org: the original freq of the signal
    :param freq_desired: the desired freq of the signal
    :return: the index of the signal with freq=freq_desired
    """
    return int(index * freq_desired / freq_org)

def rrprocess(n, f1=0.1, f2=0.25, c1=0.01, c2=0.01, lfhfratio=0.5, hrmean=60, hrstd=1, sf=512):
    """
    GENERATE RR PROCESS
    details in "MCSHARRY et al.: DYNAMICAL MODEL FOR GENERATING SYNTHETIC ELECTROCARDIOGRAM SIGNALS"
    :param flo: low frequency (Mayer wave)
    :param fhi: high frequency (RSA wave)
    :param flostd: 
    :param fhistd:
    :param lfhfratio:
    :param hrmean:
    :param hrstd:
    :param sf:
    :param n:
    :return:
    """
    # Step 1: Calculate the power spectrum:
    sig2 = 1.0
    sig1 = lfhfratio
    # Convert freq to rad/sec
    # f1 = 2.0 * math.pi * f1
    # f2 = 2.0 * math.pi * f2
    # c1 = 2.0 * math.pi * c1
    # c2 = 2.0 * math.pi * c2
    df = sf / float(n) # n is cardiac_cycle_len
    # f = np.array([2.0 * math.pi * df * i for i in range(n)])
    #f = np.linspace(0, 2.0 * math.pi, 512)
    f = np.linspace(0, 0.5, n)
    # Calaculate Power Spectrum S(f)
    # S_F = sig1 * norm.pdf(x=f, loc=f1, scale=c1) + sig2 * norm.pdf(x=f, loc=f2, scale=c2)
    # x_axis = np.linspace(0, 2.0, n)
    # plt.figure()
    # plt.plot(x_axis, S_F)
    # plt.xlabel("freq")
    # plt.ylabel("Power")
    # plt.title("Power spectrum")
    # plt.show()

    # Step 2: Create the RR-interval time series:
    # amplitudes = np.sqrt(S_F)
    amplitudes = np.linspace(0, 1, n)
    # phases = np.random.normal(loc=0, scale=1, size=len(S_F)) * 2 * np.pi
    phases = np.linspace(0, 1, n) * 2 * np.pi
    complex_series = [complex(amplitudes[i] * np.cos(phases[i]), amplitudes[i] * np.sin(phases[i])) for i in
                      range(len(phases))]
    import cmath
    # print(cmath.polar(complex_series[0]))

    T = np.fft.ifft(complex_series, n)
    T = T.real

    rrmean = 60.0 / hrmean
    rrstd = 60.0 * hrstd / (hrmean * hrmean)

    std = np.std(T)
    ratio = rrstd / std
    T = ratio * T
    T = T + rrmean
    return T


def generate_omega_function(f1, f2, c1, c2):
    """
    input: the mean and standard deviation of the low frequency and high frequency components
    return: the angular frequency of each cardiac cycle (one piece of ECG), hence we know the length of the ECG
    """
    cardiac_cycle_len = 216
    rr = rrprocess(cardiac_cycle_len,f1, f2, c1, c2, lfhfratio=0.5, hrmean=60, hrstd=1, sf=512)

    return rr


def scale_signal(signal, min_val=-0.4, max_val=1.2):
    """

    :param min:
    :param max:
    :return:
    """
    # Scale signal to lie between -0.4 and 1.2 mV :
    zmin = min(signal)
    zmax = max(signal)
    zrange = zmax - zmin

    scaled = [(z - zmin) * max_val / zrange - min_val for z in signal]

    return scaled


def smooth(x, window_len=11, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # return y
    return y[(int(window_len/2)-1):-(int(window_len/2) + 1)]


if __name__ == "__main__":
    f1 = 0.1
    f2 = 0.25
    c1 = 0.01
    c2 = 0.01
    # generate_omega_function(f1, f2, c1, c2)
    rr=generate_omega_function(f1, f2, c1, c2)
    print(rr)
    plt.plot(rr[:-1])
    plt.show()
