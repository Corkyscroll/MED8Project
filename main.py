from madmom.audio.signal import Stream, FramedSignal
from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
from madmom.features.notes import NotePeakPickingProcessor
from madmom.audio.filters import hz2midi, log_frequencies
from madmom.audio.spectrogram import Spectrogram
from madmom.audio.stft import stft, ShortTimeFourierTransform
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def tuning_frequency(spectrogram, bin_frequencies=None, num_hist_bins=15, fref=440):
    from scipy.ndimage.filters import maximum_filter

    max_spec = maximum_filter(spectrogram, size=[1,3])
    max_spec = spectrogram * (spectrogram == max_spec)

    if bin_frequencies is None:
        bin_frequencies = max_spec.bin_frequencies

    semitone_int = np.round(hz2midi(bin_frequencies, fref=fref))
    semitone_dev = semitone_int - np.round(semitone_int)
    offset = 0.5 / num_hist_bins
    hist_bins = np.linspace(-0.5 - offset, 0.5 + offset, num_hist_bins + 1)
    histogram = np.histogram(semitone_dev, weights=np.sum(max_spec, axis=0), bins=hist_bins)

    dev_bins = (histogram[1][:-1] + histogram[1][1:]) / 2.    

    dev = dev_bins[np.argmax(histogram[0])]
    return fref * 2. ** (dev / 12.)


stream = Stream(sample_rate=44100, num_channels=1)
proc = OnsetPeakPickingProcessor(online=True, threshold=0.9)
act = RNNOnsetProcessor(online=True)

frequency_range = log_frequencies(bands_per_octave=12,fmin=8,fmax=5000)

for s in stream:
    r = act(s)
    onset = proc.process_online(r)
    if len(onset) > 0:
        print(s.shape)
        fs = FramedSignal(s)
        stft = ShortTimeFourierTransform(fs)
        spec = stft.spec()
        # plt.figure()
        # plt.plot(spec)
        # plt.show()
    
        print(spec.max())