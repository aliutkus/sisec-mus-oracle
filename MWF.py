import dsdtools
import numpy as np
import itertools
from scipy.signal import stft, istft

def invert(M,eps):
    """"inverting matrices M (matrices are the two last dimensions).
    This is assuming that these are 2x2 matrices, using the explicit
    inversion formula available in that case."""
    invDet = 1.0/(eps +  M[...,0,0]*M[...,1,1] - M[...,0,1]*M[...,1,0])
    invM = np.zeros(M.shape,dtype='complex')
    invM[...,0,0] =  invDet*M[...,1,1]
    invM[...,1,0] = -invDet*M[...,1,0]
    invM[...,0,1] = -invDet*M[...,0,1]
    invM[...,1,1] =  invDet*M[...,0,0]
    return invM

def MWF(track):
    """Multichannel Wiener Filter:
    processing all channels jointly with the ideal multichannel filter
    based on the local gaussian model, assuming time invariant spatial
    covariance matrix."""

    # to avoid dividing by zero
    eps = np.finfo(np.float).eps

    #parameters for STFT
    nfft = 2048
    hop = 1024

    # compute STFT of Mixture
    N = track.audio.shape[0] #remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I,F,T)=X.shape

    # Allocate variables P: PSD, R: Spatial Covarianc Matrices
    P = {}
    R = {}
    useful_sources = ['bass','drums','other','vocals']
    J = len(useful_sources)

    for (target_name, target) in list(track.targets.items()):
        if target_name not in useful_sources: continue

        # compute STFT of target source
        Yj = stft(target.audio.T, nperseg=nfft)[-1]

        # Learn Power Spectral Density and spatial covariance matrix
        #-----------------------------------------------------------

        # 1/ compute observed covariance for source
        Rjj = np.zeros((F,T,I,I),dtype='complex')
        for (i1,i2) in itertools.product(range(I),range(I)):
            Rjj[...,i1,i2] = Yj[i1,...]*np.conj(Yj[i2,...])

        # 2/ compute first naive estimate of the source spectrogram as the
        #    average of spectrogram over channels
        P[target_name] = np.mean(np.abs(Yj)**2,axis=0)

        # 3/ take the spatial covariance matrix as the average of
        #    the observed Rjj weighted Rjj by 1/Pj. This is because the
        #    covariance is modeled as Pj Rj
        R[target_name] = np.mean(Rjj / (eps+P[target_name][...,None,None]), axis = 1)

        # add some regularization to this estimate: normalize and add small
        # identify matrix, so we are sure it behaves well numerically.
        R[target_name] = R[target_name] * I/ np.trace(R[target_name]) + eps * np.tile(np.eye(I,dtype='complex64')[None,...],(F,1,1))

        # 4/ Now refine the power spectral density estimate. This is to better
        #    estimate the PSD in case the source has some correlations between
        #    channels.

        #    invert Rj
        Rj_inv = invert(R[target_name],eps)

        #    now compute the PSD
        P[target_name]=0
        for (i1,i2) in itertools.product(range(I),range(I)):
            P[target_name] +=  1./I*np.real(Rj_inv[:,i1,i2][:,None]*Rjj[...,i2,i1])


    # All parameters are estimated. compute the mix covariance matrix as
    # the sum of the sources covariances.
    Cxx = 0
    for key in useful_sources:
        Cxx += P[key][...,None,None]*R[key][:,None,...]
    # we need its inverse for computing the Wiener filter
    invCxx = invert(Cxx,eps)

    #now separate sources
    estimates = {}
    accompaniment_source = 0
    for target_name in useful_sources:
        # computes multichannel Wiener gain as Pj Rj invCxx
        G = np.zeros(invCxx.shape,dtype='complex64')
        SR = P[target_name][...,None,None]*R[target_name][:,None,...]
        for (i1,i2,i3) in itertools.product(range(I),range(I),range(I)):
            G[...,i1,i2] += SR[...,i1,i3]*invCxx[...,i3,i2]
        SR = 0 #free memory

        # separates by (matrix-)multiplying this gain with the mix.
        Yj=0
        for i in range(I):
            Yj+=G[...,i]*X[i,...,None]
        Yj = np.rollaxis(Yj,-1) #gets channels back in first position

        # inverte to time domain
        target_estimate = istft(Yj)[1].T[:N,:]

        # take this as the source estimate
        estimates[target_name] = target_estimate

        # accumulate to the accompaniment if this source is not vocals.
        if target_name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment']=accompaniment_source
    return estimates

# initiate dsdtools
dsd = dsdtools.DB()

dsd.run(
    MWF,
    estimates_dir='MWF',
    parallel=True,
    cpus=4
)
