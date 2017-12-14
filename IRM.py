import dsdtools
import numpy as np
import functools
from scipy.signal import stft, istft

def IRM(track,alpha=2):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""

    #STFT parameters
    nfft = 2048
    hop = 1024

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0] #remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I,F,T)=X.shape

    # define the target sources
    useful_sources = ['bass','drums','other','vocals']
    J = len(useful_sources)

    # Compute sources spectrograms
    P = {}
    for j,(target_name, target) in enumerate(list(track.targets.items())):
        if target_name not in useful_sources: continue

        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        P[target_name] = np.abs(stft(target.audio.T, nperseg=nfft)[-1])**alpha

    # compute model as the sum of spectrograms
    model = eps
    for source_name in useful_sources:
        model += P[source_name]

    #now performs separation
    estimates = {}
    accompaniment_source = 0
    for j,target_name in enumerate(useful_sources):
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(np.abs(P[target_name]),model )

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = istft(Yj)[1].T[:N,:]

        #set this as the source estimate
        estimates[target_name] = target_estimate
        if target_name != 'vocals':
            #also add it to the accompaniment (=bass+drums+other)
            accompaniment_source += target_estimate
    estimates['accompaniment']=accompaniment_source

    return estimates

# initiate dsdtools
dsd = dsdtools.DB()

#default parameters
alpha = 2 #exponent for the ratio mask, take power spectrogram

dsd.run(
    functools.partial(IRM, alpha=alpha),
    estimates_dir='IRM_alpha=%d'%alpha,
    parallel=True,
    cpus=4
)
