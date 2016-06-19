# PyCNN: Cellular Neural Networks Image Processing Python Library

**Cellular Neural Networks (CNN)** are a parallel computing paradigm similar to neural networks, with the difference that communication is allowed between neighboring units only. Image Processing is one of its application. CNN processors were designed to perform image processing; specifically, the original application of CNN processors was to perform real-time ultra-high frame-rate (>10,000 frame/s) processing unachievable by digital processors.

This python library is the implementation of CNN for the application of **Image Processing**.

![alt text](http://www.isiweb.ee.ethz.ch/haenggi/CNN_web/CNN_figures/blockdiagram.gif "CNN Architecture")

## Motivation

This is an extension of a demo at 14th Cellular Nanoscale Networks and Applications (CNNA) Conference 2014. I have written a blog post, available at [Image Processing in CNN with Python on Raspberry Pi](http://blog.ankitaggarwal.me/technology/image-processing-with-cellular-neural-networks-using-python-on-raspberry-pi/).

## Dependencies
The python (2.7.6) modules are needed in order to use this library.
```
PIL (Image): 1.1.7
Scipy: 0.14.1rc1
Numpy: 1.8.1
```
Note: The module versions mentioned are tested and expected to work. The library might work on later versions, but that hasn't been tested. If you have information regarding this, please consider adding the same here. Thanks.

## Usage
*Image Processing* using CNN is simple using this library, just clone the repository and use the following code.
```python
from cnnimg import cnnimg

cnn = cnnimg()

cnn.edgedetection('input.bmp', 'output1.png')
cnn.grayscaleedgedetection('input.bmp', 'output2.png')
cnn.cornerdetection('input.bmp', 'output3.png')
cnn.diagonallinedetection('input.bmp', 'output4.png')
cnn.inversion('input.bmp', 'output5.png')
cnn.generaltemplates('input.bmp', 'output6.png')
```
#### OR
Use example.py available with the repository.
```sh
$ python example.py
```

## Example results

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/input.bmp)
*Input: input.bmp*

**Edge Detection:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output1.png)
*Output: output1.png*


**Corner Detection:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output3.png)
*Output: output3.png*


**Diagonal line Detection:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output4.png)
*Output: output4.png*


**Inversion (Logic NOT):**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output5.png)
*Output: output5.png*

## Another example (Lenna)

Here, the input is the popular face in image processing field, Lenna.

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna.gif)
*Input: lenna.gif*


**Edge Detection:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna_edge.png)
*Output: lenna_edge.png*


**Diagonal line Detection:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna_diagonal.png)
*Output: lenna_diagonal.png*


## API
```python
from cnnimg import cnnimg
```
Import the module in your main file.

```python
cnn = cnnimg()
```
Initialize the cnn class

#### cnn.edgedetection(inputimagelocation, outputimagelocation)
Function for edge detection using CNN on a given image.
#### cnn.grayscaleedgedetection(inputimagelocation, outputimagelocation)
Function for grayscale edge detection using CNN on a given image.
#### cnn.cornerdetection(inputimagelocation, outputimagelocation)
Function for corner detection using CNN on a given image.
#### cnn.diagonallinedetection(inputimagelocation, outputimagelocation)
Function for diagonal line detection using CNN on a given image.
#### cnn.inversion(inputimagelocation, outputimagelocation)
Function for invert an image using CNN.
#### cnn.generaltemplates(inputimagelocation, outputimagelocation)
Function for applying general CNN templates on a given image.

#### inputimagelocation is the location of the input image, Type: String.
#### outputimagelocation is the location of the output image, Type: String.


## Contributors

#### Author: Ankit Aggarwal

If anybody is interested in working on developing this library, fork and feel free to get in touch with me.

## License

[MIT License](https://github.com/ankitaggarwal011/CNN-Image-Processing/blob/master/LICENSE)
