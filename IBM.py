import stft
import dsdtools
import numpy as np


def IBM(track):
    # get the audio mixture as numpy array shape=(nun_sampl, 2)
    theta = 0.5
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    M = stft.spectrogram(
        track.audio, framelength=1024, hopsize=512, halved=True
    )

    estimates = {}
    for target_name, target in list(track.targets.items()):
        print(target_name)

        # compute STFT of target source
        X = stft.spectrogram(
            target.audio, framelength=1024, hopsize=512, halved=True
        )

        # Create Binary Mask
        Mask = np.divide(np.abs(M), (eps + np.abs(X)))
        bg = np.where(Mask >= theta)
        sm = np.where(Mask < theta)
        Mask[bg[0], bg[1]] = 1.
        Mask[sm[0], sm[1]] = 0.

        # multiply mask
        Y = np.multiply(M, Mask)

        # inverte to time domain
        target_estimate = stft.ispectrogram(
            Y, framelength=1024, hopsize=512, halved=True
        )

        estimates[target_name] = target_estimate

    return estimates

# initiate dsdtools
dsd = dsdtools.DB()

dsd.run(
    IBM,
    estimates_dir='./IBM',
)
