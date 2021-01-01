import numpy as np
from scipy import signal
from scipy import fft
import matplotlib.pyplot as plt

__author__ = "Guo, Jiangling"
__email__ = "tguojiangling@jnu.edu.cn"
__version__ = "20201224.2115"

# Common sequence generators.
seq_delta = lambda n_min, n_max: (np.array([ 1 if n==0 else 0 for n in range(n_min, n_max+1)]), 0 - n_min)
seq_u = lambda n_min, n_max: (np.array([ 0 if n<0 else 1 for n in range(n_min, n_max+1)]), 0-n_min)
seq_R_N = lambda N, n_min, n_max: (np.array([ 0 if (n<0 or n>=N) else 1 for n in range(n_min, n_max+1)]), 0-n_min)
seq_real_exp = lambda a, n_min, n_max: (np.array([ 0 if n<0 else a**n for n in range(n_min, n_max+1)]), 0-n_min)
seq_complex_exp = lambda sigma, omega_0, n_min, n_max: (np.array([ np.exp((sigma + 1j*omega_0)*n) for n in range(n_min, n_max+1)]), 0-n_min)
seq_sin = lambda A, omega_0, phi, n_min, n_max: (np.array([ A*np.sin(omega_0*n + phi) for n in range(n_min, n_max+1)]), 0-n_min)

# Some useful functions.
mse = lambda x, y: np.sum(np.abs(x - y)**2)/len(x)
filter_len = lambda b, a: max(1 if np.isscalar(b) else len(b), 1 if np.isscalar(a) else len(a))

# A selection of tick format functions that can be used by `analyze_filter`.
def tick_format_rad_to_pi(value, tick_number):
    """Convert value (in rad) to multiple of pi."""
    return ('%.2f' % (value/np.pi)).rstrip('0').rstrip('.') + '$\pi$'

def tick_format_pi(value, tick_number):
    return ('%.2f' % (value/np.pi)).rstrip('0').rstrip('.') + '$\pi$'

def tick_format_append_pi(value, tick_number):
    """Append pi symbol to the value."""
    return ('%.2f' % (value)).rstrip('0').rstrip('.') + '$\pi$'

def tick_format_append_hz(value, tick_number):
    """Append Hz unit to the value."""
    return ('%.2f' % (value)).rstrip('0').rstrip('.') + 'Hz'

def tick_format_to_khz(value, tick_number):
    """Show the value in kHz"""
    return ('%.2f' % (value/1000)).rstrip('0').rstrip('.') + 'kHz'

def analyze_filter(bands, b, a=1, show_plot=False, samples_per_band=129, fs=2, tick_format=tick_format_append_pi, amp_in_dB=True):
    """Find the R_p and A_s of the given filter and optionally show the frequency response plots.
    
    Parameters:
      bands : array_like
        A list of tuples, `(band_type, band_start, band_end)`, to describe each band.
        `band_type`: str, one of 'pass', 'tran', 'stop'
        `band_start`, `band_end`: float, the start and end frequencies, normalized to `fs`.
      b : array_like
        Numerator of the transfer function.
      a : array_like, optional
        Denominator of the transfer function. Default is 1 (FIR filter).
      show_plot : bool, optional
        Set to True to show the frequency response plots. Default is False.
      samples_per_band : int, optional
        Number of frequency samples per band (pass, transition and stop). Generally, more samples will give more accurate results. Default is 129.
      fs: float, optional
        Sampling frequency. Default is 2. (Tips: set to 2*numpy.pi for representing frequencies in rad/sample.)
      tick_format: function, optional
        Function to format tick values on x-axis. Default is `tick_format_append_pi`, which append a pi symbol to the values.
      amp_in_dB: bool, optional
        Whether the amplitude is represented in dB or not. Default is True.
    
    Returns:
      R_p: float
        Pass band ripple, in dB if amp_in_dB is True.
      A_s: float
        Stop band attenuation, in dB if amp_in_dB is True.
    """

    if amp_in_dB:
      dB = lambda x: 20 * np.log10(x)
      amp_unit = '(dB)'
    else:
      dB = lambda x: x
      amp_unit = ''

    NUM_BANDS = len(bands)
    [BAND_TYPE, BAND_START, BAND_END] = range(3)

    # Compute frequency respone samples for each band.
    w = []      # To store frequency samples for each band.
    H = []      # To store frequency respone samples for each band.
    for (band_type, w_start, w_end) in bands:
        w_tmp, H_tmp = signal.freqz(
          b=b, a=a, 
          worN=np.linspace(w_start, w_end, samples_per_band), 
          fs=fs)
        w.append(w_tmp)
        H.append(H_tmp)

    # Normalize |H| to 1.     
    H_abs = np.abs(H)
    H_norm_factor = np.max(H_abs)
    H = H / H_norm_factor
    H_abs = H_abs / H_norm_factor

    # Find the minimum pass band ripple and maximum stop band attenuation (across all respective bands).
    pass_min = 1
    stop_max = 0
    for band in range(NUM_BANDS):
      band_type = bands[band][BAND_TYPE]
      if band_type=='pass':
        pass_min = min(pass_min, np.min(H_abs[band]))
      elif band_type=='stop':
        stop_max = max(stop_max, np.max(H_abs[band]))
    pass_min, stop_max = dB(pass_min), dB(stop_max)
    
    if show_plot:
        NUM_PLOTS = 2
        [AMP_PLOT, PHASE_PLOT] = range(NUM_PLOTS)
        BAND_COLOR = {'pass':'green', 'tran':'blue', 'stop':'red'}

        fig, axs = plt.subplots(NUM_PLOTS, 1 ,sharex='col')

        # Plot the frequency response.
        for band in range(NUM_BANDS):
            band_type = bands[band][BAND_TYPE]
            axs[AMP_PLOT].plot(w[band], dB(H_abs[band]), color=BAND_COLOR[band_type])
            axs[PHASE_PLOT].plot(w[band], np.angle(H[band]), color=BAND_COLOR[band_type])
        axs[AMP_PLOT].legend([bands[band][0] for band in range(3)])
        axs[PHASE_PLOT].set_xlim(left=0, right=fs/2)
        axs[AMP_PLOT].set_ylabel('Amplitude' + amp_unit)
        axs[PHASE_PLOT].set_ylabel('Phase')
        axs[PHASE_PLOT].set_xlabel('Frequency')
        axs[AMP_PLOT].grid()
        axs[PHASE_PLOT].grid()

        # Set the tick format
        axs[PHASE_PLOT].xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        axs[PHASE_PLOT].xaxis.set_major_locator(plt.MultipleLocator(fs/8))
        axs[PHASE_PLOT].yaxis.set_major_formatter(plt.FuncFormatter(tick_format_rad_to_pi))
        axs[PHASE_PLOT].yaxis.set_major_locator(plt.MultipleLocator(np.pi/2))

        # Add horizontal lines to indicate R_p and A_s.
        axs[AMP_PLOT].axhline(y=pass_min, linestyle='--', color=BAND_COLOR['pass'])
        axs[AMP_PLOT].axhline(y=stop_max, linestyle='--', color=BAND_COLOR['stop'])
        axs[AMP_PLOT].secondary_yaxis('right').set_yticks([pass_min, stop_max])

        # Add vertical lines to indicate band egdes.
        band_edges = []
        for band in range(NUM_BANDS):
          band_type, band_start, band_end = bands[band]
          if band_type=='tran':
            for plot in range(NUM_PLOTS):
              axs[plot].axvline(band_start, linestyle='--')
              axs[plot].axvline(band_end, linestyle='--')
              band_edges.append(band_start)
              band_edges.append(band_end)
        ax_phase_top = axs[PHASE_PLOT].secondary_xaxis('top')
        ax_phase_top.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax_phase_top.set_xticks(band_edges)

    if amp_in_dB:
      R_p, A_s = -pass_min, -stop_max
    else:
      R_p, A_s = pass_min, stop_max
    return R_p, A_s
    
def plot_signal(x, fs, axs):
  NUM_PLOTS = 2
  [TIME_PLOT, FREQ_PLOT] = range(NUM_PLOTS)

  N = len(x)
  n = np.arange(N)
  t = n/fs
  axs[TIME_PLOT].plot(t, x)
  axs[TIME_PLOT].set_xlabel('t (sec)')

  f = n/N*fs  # [NOTICE] `f = n*fs/N` will overflow when n*fs is greater than 2^16.
  axs[FREQ_PLOT].plot(f, abs(fft.fft(x)))
  axs[FREQ_PLOT].set_xlabel('f (Hz)')
  axs[FREQ_PLOT].xaxis.set_major_locator(plt.MultipleLocator(fs/4))
  
def plot_signals(legends, signals, fs, axs, titles):
    for x in signals:
        f = plot_signal(x, fs, axs)
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.legend(legends)
