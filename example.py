#!/usr/bin/python
# -*- coding: utf-8 -*-
from cnnimg import cnn

# Perform respective image processing techniques on the given image

cnn.edgedetection('input.bmp', 'output1.png')
cnn.grayscaleedgedetection('input.bmp', 'output2.png')
cnn.cornerdetection('input.bmp', 'output3.png')
cnn.diagonallinedetection('input.bmp', 'output4.png')
cnn.inversion('input.bmp', 'output5.png')
cnn.generaltemplates('input.bmp', 'output6.png')
