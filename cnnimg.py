"""
The MIT License (MIT)

Copyright (c) 2014 Ankit Aggarwal <ankitaggarwal011@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

## start of module
import scipy.signal as sig
import scipy.integrate as sint
import Image as img
import numpy as np
import os.path
import warnings

#Ignore warnings
warnings.filterwarnings("ignore")

##Supported Filetypes
filetypes = ["jpeg", "jpg", "png", "tiff", "gif", "bmp"]

##Helper Functions start
def f(x, t):
    x = x.reshape((n,m))
    dx = -x + Ib + Bu + sig.convolve2d(cnn(x),tempA,'same')
    return dx.reshape(m*n)

def cnn(x):
  return 0.5*(abs(x + 1) - abs(x - 1))

def isvalid(inputlocation):
    if not os.path.isfile(inputlocation): 
        raise Exception("File does not exist")
    elif inputlocation.split(".")[1].lower() not in filetypes or "." not in inputlocation:
        raise Exception("Invalid File")
    else:
        return True

def imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t):
    global n, m, Bu
    gray = img.open(inputlocation).convert('RGB')
    m,n = gray.size
    u = np.array(gray)
    u=u[:,:,0]
    z0 = (u)*initialcondition
    Bu = sig.convolve2d(u,tempB,'same')
    z0 = z0.reshape(z0.size)
    z =cnn(sint.odeint(f, z0, t))
    l = z[z.shape[0]-1,:].reshape((n,m))
    l = np.flipud(l/(255.0))
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            l[i][j] = np.uint8(round(l[i][j]*255))
    l = img.fromarray(l).convert('RGB')
    l.save(outputlocation)
    print "Image Processing is successful."
## Helper Functions end

## Image Processing methods start    

#edge detection #working
def edgedetection(inputlocation = "", outputlocation = "output.png"):
    if not isvalid(inputlocation):
        exit()
    global Ib, tempA
    tempA = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    tempB = np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
    Ib = -1.0
    t = np.linspace(0, 10.0, num=2)
    initialcondition = 0.0
    imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)

#grayscale edge detection #working
def grayscaleedgedetection(inputlocation = "", outputlocation = "output.png"):
    if not isvalid(inputlocation):
        exit()
    global Ib, tempA
    tempA = np.array([[0.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]]) 
    tempB = np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
    Ib = -0.5
    t = np.linspace(0, 1.0, num=100)
    initialcondition = 0.0
    imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)

#corner #working
def cornerdetection(inputlocation = "", outputlocation = "output.png"):
    if not isvalid(inputlocation):
        exit()
    global Ib, tempA
    tempA = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]) 
    tempB = np.array([[-1.0,-1.0,-1.0],[-1.0,4.0,-1.0],[-1.0,-1.0,-1.0]])  
    Ib = -5.0
    t = np.linspace(0, 10.0, num=10)
    initialcondition = 0.0
    imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)

#diagonal line detector #working
def diagonallinedetection(inputlocation = "", outputlocation = "output.png"):
    if not isvalid(inputlocation):
        exit()
    global Ib, tempA
    tempA = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]) 
    tempB = np.array([[-1.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,-1.0]]) 
    Ib = -4.0
    t = np.linspace(0, 0.2, num=100)
    initialcondition = 0.0
    imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)

# logic NOT #working
def logicNOT(inputlocation = "", outputlocation = "output.png"):
    if not isvalid(inputlocation):
        exit()
    global Ib, tempA
    tempA = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]) 
    tempB = np.array([[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]) 
    Ib = -2.0
    t = np.linspace(0,10.0,num=100)
    initialcondition = 0.0
    imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
## Image Processing methods end

## General Image Processing method with template input
def generaltemplates(inputlocation = "", outputlocation = "output.png", tempA_A = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]], tempB_B = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]], initialcondition = 0.0, Ib_b = 0.0, t = np.linspace(0, 10.0, num=2)):
    if not isvalid(inputlocation):
        exit()
    global Ib, tempA
    tempA=np.array(tempA_A)
    tempB=np.array(tempB_B)
    Ib = Ib_b
    imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)

## end of module
