import dsdtools
import numpy as np
import functools
from scipy.signal import stft, istft

def IBM(track,alpha=1,theta=0.5):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal binary mask.

    the mix is send to some source if the spectrogram of that source over that
    of the mix is greater than theta, when the spectrograms are take as magnitude
    of STFT raised to the power alpha. Typical parameters involve a ratio of
    magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    #parameters for STFT
    nfft = 2048
    hop = 1024

    #small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0] #remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I,F,T)=X.shape

    # define the target sources
    useful_sources = ['bass','drums','other','vocals']
    J = len(useful_sources)

    #perform separtion
    estimates = {}
    accompaniment_source = 0
    for target_name, target in list(track.targets.items()):
        if target_name not in useful_sources: continue

        # compute STFT of target source
        Yj = stft(target.audio.T, nperseg=nfft)[-1]

        # Create Binary Mask
        Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
        Mask[np.where(Mask>theta)] = 1
        Mask[np.where(Mask<theta)] = 0

        # multiply mask
        Yj = np.multiply(X, Mask)

        # inverte to time domain and set same length as original mixture
        target_estimate = istft(Yj)[1].T[:N,:]

        # set this as the source estimate
        estimates[target_name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if target_name != 'vocals':
            accompaniment_source += target_estimate
    # set accompaniment source
    estimates['accompaniment']=accompaniment_source

    return estimates

# initiate dsdtools
dsd = dsdtools.DB()

#default parameters
alpha = 1 #exponent for the ratio Mask
theta = 0.5 # threshold

dsd.run(
    functools.partial(IBM, alpha=alpha,theta=theta),
    estimates_dir='IBM',
    parallel=True,
    cpus=4
)
