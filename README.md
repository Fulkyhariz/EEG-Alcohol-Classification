# Alcoholic EEG Signal Classification

<div style="text-align: justify">
This is a project for my final year thesis, I compare two feature extraction method, that is RWB that is proposed <a href="https://ieeexplore.ieee.org/document/8168473">here</a>, and Bispectrum with filter that is proposed <a href="https://www.mdpi.com/1999-4893/10/2/63">here</a>, to classify EEG signal of an alcoholic. For more in depth explanation about the theorem you can refer to those papers.
</div>

## Environment

|     Library       |     Version    |
|-------------------|---------------|
|     TensorFlow    |     2.10.1    |
|     NumPy         |     1.5.3     |
|     Pandas        |     1.23.5    |
|     Matplotlib    |     3.6.2     |
|     Pywt          |     1.4.1     |
|     Stringray     |     1.10.1    |

<div style="text-align: justify">
For the more complete specification of the environment I use, you can refer to this <a href="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Environment.yaml">yaml</a> file, you can also directly import it to create new anaconda environment
</div>

## Dataset

<div style="text-align: justify">
The dataset is obtained from <a href="https://archive.ics.uci.edu/dataset/121/eeg+database">UCI Machine Learning repository</a>, there are three variation of the dataset and The Large Dataset version will be used in this project.
</div>

|Dataset Version|Number of Subject|Number of Alcoholic Subject|Number of Control Subject|Number of Trial|File Size|
|--------------------------------|----------------------------|--------------------------------------|------------------------------------|--------------------------|--------------------|
|The Small Data Set|2|1|1|3|3,9 MB|
|The Large Data Set|20|10|10|30 |75,5 MB|
|The Full Dataset |122|65|57|120|685 MB|

## Method
<p align="justify">
The are two method that is used to extract the feature of the data, that is RWB (Relative Wavelet Bispectrum) and Bispectrum-Gaussian (Bispectrum with gaussian filter for downsampling)
</p>

### RWB (Relative Wavelet Bispectrum)
<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/RWB%20Paper.png?raw=true" width = '60%'>
</p>

<p align="justify">
The autocorrelation and 3rd order cumulant matrix is calculated using the <a href="https://docs.stingray.science/notebooks/Bispectrum/bispectrum_tutorial.html">bispectrum</a> function from Stringray, and the wavelet decomposition is done by using <a href="https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html">pywavelets</a>. After the decomposition the RWB is calculated using:
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/RWB%20Formula.jpg?raw=true" width = '60%'>
</p>

<p align="justify">
Where it basically calculate the energy of each wavelet decomposition result and divide it by the total energy of all approximation and detail part of the decomposition result
</p>

### Bispectrum-Gaussian
<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Bispectrum%20paper.png?raw=true" width = '60%'>
</p>

<p align="justify">
The filter size that is used is a 5x5 2d gaussian filter, with follow the quadratic series, for example if the lag for the autocorrelation is 256 then the filter will be (128, 64, 32, 16, 16). The filter shape is:
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/filter_shape.drawio-01.png?raw=true" width = '80%'>
</p>

<p align="justify">
The filtered feature then will be pooled using average pool with pool size 5x5 following the filter size. It will looked like this:
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/bispec_feature_contour.png?raw=true" width = '60%'>
</p>

### Dimension Variation
The feature's dimension will be varied according to:
|      Dimension Variation     |      Dimension     |
|------------------------------|--------------------|
|      RWB 1D                  |      1 x 366       |
|      RWB 2D                  |      61 x 6        |

|      Dimension Variation       |      Dimension     |
|--------------------------------|--------------------|
|      Bispectrum    1D          |      1 x 1525      |
|      Bispectrum    2D          |      61 x 5 x5     |
|      Bispectrum    2D Flat     |      61 x 25       |

## Model Architechture

This project uses 2 kinds of classifier, CNN and ANN, the CNN will uses Conv1D and Conv2D according to the feature dimension variation.

### ANN Architechture:
|      Layer     |      Layer type     |      Node     |      Activation Function     |
|----------------|---------------------|---------------|------------------------------|
|      0         |      Input          |      -        |      -                       |
|      1         |      Flatten        |      -        |      -                       |
|      2         |      Dense          |      512      |      ReLU                    |
|      3         |      Dense          |      256      |      ReLU                    |
|      4         |      Dense          |      1        |      Sigmoid                 |

### CNN 1D Architechture:

|      Layer     |      Layer type               |      Filter     |      Kernel Size     |      Node     |      Activation Function     |
|----------------|-------------------------------|-----------------|----------------------|---------------|------------------------------|
|      0         |      Input                    |      -          |      -               |      -        |      -                       |
|      1         |      Convolution Layer 1D     |      16         |      3               |      -        |      ReLU                    |
|      2         |      Max Pooling Layer 1D     |      -          |      3               |      -        |                              |
|      3         |      Flatten                  |      -          |      -               |      -        |      -                       |
|      4         |      Dense                    |      -          |      -               |      512      |      ReLU                    |
|      5         |      Dense                    |      -          |      -               |      256      |      ReLU                    |
|      6         |      Dense                    |      -          |      -               |      1        |      Sigmoid                 |

### CNN 2D Architechture:

|      Layer     |      Layer type               |      Filter     |      Kernel Size     |      Node     |      Activation Function     |
|----------------|-------------------------------|-----------------|----------------------|---------------|------------------------------|
|      0         |      Input                    |      -          |      -               |      -        |      -                       |
|      1         |      Convolution Layer 2D     |      16         |      3 x 3           |      -        |      ReLU                    |
|      2         |      Max Pooling Layer 2D     |      -          |      3 x 3           |      -        |                              |
|      3         |      Flatten                  |      -          |      -               |      -        |      -                       |
|      4         |      Dense                    |      -          |      -               |      512      |      ReLU                    |
|      5         |      Dense                    |      -          |      -               |      256      |      ReLU                    |
|      6         |      Dense                    |      -          |      -               |      1        |      Sigmoid                 |

## Result

The model evaluation is using 2-fold cross validation as seen in this image:

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/cross%20validation.drawio.png?raw=true" width = '80%'>
</p>

### Comparing CNN and ANN in RWB method

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Table%20RWB.jpg?raw=true" width = '80%'>
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Canvas-01.png?raw=true" width = '80%'>
</p>

### Comparing CNN and ANN in Bispectrum-Gaussian method

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Table%20Bispec.jpg?raw=true" width = '80%'>
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Canvas-02.png?raw=true" width = '80%'>
</p>

### Comparing Accuracy of RWB and Bispectrum method

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Table%20RWB%20Bispectrum.jpg?raw=true" width = '80%'>
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Canvas-03.png?raw=true" width = '80%'>
</p>

### Comparing the Accuracy of the Lag Varation of RWB method

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Table%20LAG.jpg?raw=true" width = '80%'>
</p>

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Canvas-04.png?raw=true" width = '80%'>
</p>

From this comparison we can imply that, for EEG Alcoholic signal classification, the RWB method is better than Bispectrum-Gaussian in terms of overall acurracy, and also the higher the lag value, the more information its contains, and the higher the accuracy.