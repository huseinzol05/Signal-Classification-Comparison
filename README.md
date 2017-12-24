# Classify-Sound-Tensorflow
Classify sound using Deep Learning on Tensorflow and various Gradient Boosting techniques.

Use spectrogram, Extract Derivative Features, Log Mel Frequency Energy and MFCC for sound featuring.

### You can download the dataset [here from urban sound](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html), also [here from google](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)

All training session using these parameters
```python
learning_rate = 0.001
epoch = 20
batch_size = 128
```

for convolutional parameters
```python
sound_dimension = [64, 512]
```

for recurrent parameters
```python
time_stamp = 64
dimension = 512
```

## Spectogram Urban Sound
feed-forward neural network
```text
testing accuracy: 0.463215
                  precision    recall  f1-score   support

 air_conditioner       0.41      0.32      0.36       105
        car_horn       0.46      0.40      0.43        58
children_playing       0.33      0.12      0.17       126
        dog_bark       0.67      0.58      0.62       134
        drilling       0.42      0.72      0.53       137
   engine_idling       0.54      0.30      0.39       127
        gun_shot       0.60      0.59      0.60        59
      jackhammer       0.36      0.73      0.49       122
           siren       0.57      0.74      0.64       114
    street_music       0.31      0.13      0.19       119

     avg / total       0.46      0.46      0.44      1101
```

convolutional
```text
testing accuracy: 0.4
                  precision    recall  f1-score   support

 air_conditioner       0.40      0.20      0.27        10
        car_horn       0.29      0.33      0.31         6
children_playing       0.16      0.33      0.21         9
        dog_bark       0.36      0.31      0.33        16
        drilling       0.50      0.55      0.52        11
   engine_idling       0.56      0.56      0.56         9
        gun_shot       0.33      0.33      0.33         3
      jackhammer       0.57      0.44      0.50         9
           siren       0.46      0.67      0.55         9
    street_music       0.67      0.31      0.42        13

     avg / total       0.45      0.40      0.40        95
```

recurrent
```text
testing accuracy: 0.453488
                  precision    recall  f1-score   support

 air_conditioner       0.35      0.55      0.43        11
        car_horn       0.50      0.40      0.44         5
children_playing       0.20      0.11      0.14         9
        dog_bark       1.00      0.64      0.78        11
        drilling       0.38      0.45      0.42        11
   engine_idling       1.00      0.44      0.62         9
        gun_shot       1.00      0.50      0.67         2
      jackhammer       0.50      0.60      0.55        10
           siren       0.50      0.22      0.31         9
    street_music       0.26      0.56      0.36         9

     avg / total       0.54      0.45      0.46        86
```
