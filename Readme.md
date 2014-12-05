Author: Ankit Aggarwal
Contact: ankitaggarwal011@gmail.com

Cellular Neural Networks Image Processing Python Library:

Cellular Neural Networks (CNN) are a parallel computing paradigm similar to neural networks, with the difference that communication is allowed between neighbouring units only. Image Processing is one of its application. CNN processors were designed to perform image processing; specifically, the original application of CNN processors was to perform real-time ultra-high frame-rate (>10,000 frame/s) processing unachievable by digital processors.

This python library is the implementation of CNN for the application of Image Processing. This is an extension of a demo at 14th Cellular Nanoscale Networks and Applications (CNNA) Conference 2014.

Dependencies:
Image
Scipy
Numpy

Usuage:
import cnnimage as cnn
cnn.edgedetection(inputimagelocation, outputimagelocation)
cnn.grayscaleedgedetection(inputimagelocation, outputimagelocation)
cnn.cornerdetection(inputimagelocation, outputimagelocation)
cnn.diagonallinedetection(inputimagelocation, outputimagelocation)
cnn.logicNOT(inputimagelocation, outputimagelocation)
cnn.generaltemplates(inputimagelocation, outputimagelocation)

inputimagelocation is the location of the input image, Type: String.
outputimagelocation is the location of the output image, Type: String.

See example.py for more details.

If anybody is interested in working on developing this library, fork and feel free to get in touch with me.