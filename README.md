# Neural Network based Source Separation

using a deep neural network jointly compute source separation results

## Installation

```
brew install libsndfile python
pip install -r requirements.txt
```

## Usage

### Training

Start training by call
```
train.py {1} -x {2} -d {3} -b {4} -e {5} -c {6} -l {7} --scale --greedy
```

where:
  * {1} is the keras model (see folder models)
  * {2} is the feature transformation (see transformers.py)
  * {3} is the dataset which defaults to ```dsd```
  * {4} is the batch size
  * {5} is the number of epochs
  * {6} is the context is number of samples C=2 means [i-2, i-1, i, i+1, i+2]
  * {7} the number of fully connected layers, defaults to 3
  * `--scale` turns on input scaling using var=1
  * `--norm` enables input normalisation

```
python train.py mlp -x STFTC STFTC -d dsd -b 64 -e 3 -c 3 3 --scale
```

## Comparison

### STFT

Context size = 2, results in shape (nb_samples, 5, 513), reshaped to array

```
X shape: (None, 2565)
Y shape: (None, 1026)
```

```
python train.py mlp -d dsd -x STFTC -b 32 -c 2 -e 30 --scale
```

### GFT and CFT

```
framelength=1024,
hopsize=512,
W=(10, 5),
mhop=(10, 2)
```

Context size = 0, because GFT and CFT have already context

```
X shape: (None, 2565)
Y shape: (None, 1026)
```

```
python train.py mlp -d dsd -x CFT -b 32 -c 0 -e 30 --scale
python train.py mlp -d dsd -x CFT -b 32 -c 0 -e 30 --scale
```
