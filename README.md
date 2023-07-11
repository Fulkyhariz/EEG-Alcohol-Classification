# Alcoholic EEG Signal Classification

<div style="text-align: justify">
This is a project for my final year thesis, I compare two feature extraction method, that is RWB that is proposed <a href="https://ieeexplore.ieee.org/document/8168473">here</a>, and Bispectrum with filter that is proposed <a href="https://www.mdpi.com/1999-4893/10/2/63">here</a>, to classify EEG signal of an alcoholic. For more in depth explanation about the theorem you can refer to those papers.
</div>

## Environment

<p align="center">
    <img src="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Image/Environment.jpg?raw=true" width = '60%'>
</p>

<div style="text-align: justify">
For the more complete specification of the environment I use, you can refer to this <a href="https://github.com/Fulkyhariz/EEG-Alcohol-Classification/blob/main/Environment.yaml">yaml</a> file, you can also directly import it to create new anaconda environment
</div>

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