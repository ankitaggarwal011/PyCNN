#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycnn import pycnn

# Initialize the cnn class
cnn = pycnn()

# Perform respective image processing techniques on the given image

cnn.edgedetection('images/lenna.gif', 'images/lenna_edge.png')
cnn.diagonallinedetection('images/lenna.gif', 'images/lenna_diagonal.png')
