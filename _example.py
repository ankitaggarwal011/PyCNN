#!/usr/bin/python
# -*- coding: utf-8 -*-
from cnnimg import cnn

# Perform respective image processing techniques on the given image

cnn.edgedetection('input.gif', '_output1.png')
cnn.grayscaleedgedetection('input.gif', '_output2.png')
cnn.cornerdetection('input.gif', '_output3.png')
cnn.diagonallinedetection('input.gif', '_output4.png')
cnn.inversion('input.gif', '_output5.png')
