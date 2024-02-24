# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:43:27 2024

@author: tsing

DSP Tools v02

This is a set of tools for analyzing and synthesizing samples of waveforms
using the Discrete Fourier Transform.
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt

"""
--------------------------------------------
PHASE I - Conversion to Frequency Domain
--------------------------------------------
"""
def M1_1(sample, window):
    Windowed_Sample = np.multiply(sample, window)                               #Multiplies each sample by its corresponding window coefficient
    return Windowed_Sample

def load_sample(filename):
    sample = np.loadtxt(filename, ndmin=2)
    return sample

def load_window(filename):
    window = np.loadtxt(filename)
    return window

def generate_window(filename, Number_of_Samples, Bias_In, Window_In, Trig=False, Segments=1):
    
    N = Number_of_Samples
    
    bias = np.swapaxes(np.array(Bias_In,ndmin=2), 0 , 1)                        #This is a list of Bias Values, for example: +0, +1, +2 would be [0,1,2]
    bias_array = np.multiply(bias, np.array_split(np.ones(N), Segments))        #This generates an N-dimensioned 1d array with the Biases mapped to each value
    
    window_matrix = np.mgrid[0:N]                                               #Generates an N-dimensioned 1d array where each value is its coordinate
    if Trig == True:
        window_matrix = -np.cos((2 * np.pi * window_matrix) / N)                #Allows trig based windows like Hamming and Hanning
    window_matrix = np.array_split(window_matrix, Segments)
    
    window_function = np.swapaxes(np.array(Window_In, ndmin=2), 0, 1)           #Generates an N-dimensional 1d array with the window functions mapped to each value
    
    window = np.reshape(np.add(np.multiply(window_function, window_matrix), bias_array), (1,-1))        # w(n) = (window_function * window_matrix) + bias
    
    savedata('window',window)
    return window

#--------------------------------------------

def M1_2(sample, DFTmatrix):                                                    # Multiplies x(n) by DFTmatrix
    raw = np.multiply(sample, DFTmatrix)
    return raw

def load_DFTmatrix(filename):
    DFTmatrix = np.loadtxt(filename)
    return DFTmatrix

def generate_DFTmatrix(filename, Number_of_Samples):
    
    N = Number_of_Samples
    u_Coeff = (2*np.pi)/N
    
    coords = np.mgrid[0:N,0:N]                                                  # N x N array [Frequency Bin, Sample Number]
    
    Real_Coordinate = np.cos(np.multiply(u_Coeff  * coords[0] , coords[1]))
    Imaginary_Coordinate = 1j * -np.sin(np.multiply(u_Coeff * coords[0], coords[1]))
    
    DFTmatrix = Real_Coordinate + Imaginary_Coordinate
    savedata(filename,DFTmatrix)
    return DFTmatrix

#--------------------------------------------

def S1(raw):                                                                    # Takes the sum of each frequency bin
    sums = np.add.reduce(raw)
    sums = np.multiply(sums, (2/N))
    return sums

def Polar_Conversion(filename, sums):                                           # Converts the sums to phasors in polar form. This is the output of the DFT.
    amplitude = np.abs(sums)
    phase_angle = np.arctan(sums.real / sums.imag)
    
    plot_DFTamplitude(amplitude)
    plot_DFTphase_angle(phase_angle)
    
    PhasorArray =np.swapaxes((amplitude, phase_angle), 0, 1)
    savedata(filename, PhasorArray)
    return PhasorArray

def load_PhasorArray(filename):
    PhasorArray = np.loadtxt(filename)
    return PhasorArray

def generate_PhasorArray(filename):                                             # **WIP** This will generate the values to synthesize a wave
    return PhasorArray

"""
--------------------------------------------
PHASE II - Coversion to Time Domain
--------------------------------------------
"""
def M2_1(PhasorArray, mask):
    masked_PhasorArray = np.multiply(PhasorArray, mask)
    return masked_PhasorArray

def load_mask(filename):
    mask = loadtxt(filename)
    return mask

def generate_mask(filename):
    return mask

#--------------------------------------------

def M2_2(PhasorArray, IDFTmatrix):
    SignalArray = np.multiply(PhasorArray, IDFTmatrix)
    return SignalArray

def load_IDFTmatrix(filename):
    IDFTmatrix = np.loadtxt(filename)
    return IDFTmatrix

def generate_IDFTmatrix(filename, Number_of_Samples):
    
    N = Number_of_Samples
    u_Coeff = (2 * np.pi) / N
    
    coords = np.mgrid[0:N,0:N]
    
    Real_Coordinate = np.cos(np.multiply(coords[0],u_Coeff * coords[1]))
    Imaginary_Coordinate = 1j * np.sin(np.multiply(coords[0],u_Coeff * coords[1]))
    
    IDFTmatrix = Real_Coordinate + Imaginary_Coordinate
    savedata(filename, IDFTmatrix)
    return IDFTmatrix

#--------------------------------------------

def S2(filename, SignalArray):
    signal = np.add.reduce(SignalArray, axis=1) / 2
    savedata(filename, signal.real)
    plot_signal(signal)
    return signal

"""
--------------------------------------------
Utility Functions
--------------------------------------------
"""

def savedata(filename, data):                                                   #Saves an array to .csv for storage and viewing
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d - %H_%M_%S')
    filetype = '.csv'
    file_out = str(filename + " --  " + timestamp + filetype)                   #I would like to add a Project Name and counter to the file names in the future
    print("Saved " + file_out)
    np.savetxt(file_out, data, delimiter = ',')
    
"""
--------------------------------------------
Graphing Functions
--------------------------------------------
"""

def save_plot(filename):
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d - %H_%M_%S')
    filetype = '.svg'
    file_out = str(filename + ' -- ' + timestamp + filetype)
    print('Saved ' + file_out)
    plt.savefig(file_out)

def plot_DFTamplitude(amplitudes):
    x = np.arange(len(amplitudes))
    y = amplitudes
    
    fig, ax = plt.subplots()
    
    ax.set_title('Calculated Amplitudes')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency Bin')
    ax.grid()
    ax.stem(x,y)
  
    filename = 'DFT Amplitudes'
    save_plot(filename)
    
 
def plot_DFTphase_angle(phase_angles):
    x = np.arange(len(phase_angles))
    y = phase_angles
    
    fig, ax = plt.subplots()
    
    ax.set_title('Calculated Phase Angles')
    ax.set_ylabel('Phase Angle')
    ax.set_xlabel('Frequency Bin')
    ax.grid()
    ax.stem(x,y)
    
    filename = 'DFT Phase Angles'
    save_plot(filename)
    
def plot_signal(signal):
    x = np.arange(len(signal))
    y = signal
    
    fig, ax = plt.subplots()
    
    ax.set_title('Calculated Signal')
    ax.set_ylabel('Signal')
    ax.set_xlabel('Sample Number')
    ax.grid()
    ax.plot(x,y)
    
    filename = 'Signal Output'
    save_plot(filename)
    
"""
--------------------------------------------
Main
--------------------------------------------
"""

'''
--------------------------------------------
PHASE 1 Conversion to Frequency Domain
--------------------------------------------

This section converts an array of input samples in
the time domain and converts them into phasors with
amplitude and phase angle.
'''

#--------------------------------------------
# DFT Input Setup
#--------------------------------------------

#N=64
sample_file = 'Golden Arches.csv'

sample = load_sample(sample_file)                                               #This loads a sample for analysis
N = len(sample)
print(N , " Samples")


#--------------------------------------------
# Window Function Setup
#--------------------------------------------

window_file = 'window'
bias = [1]
windows = [0]

window = generate_window(window_file,N,bias,windows,Trig=False, Segments=1)

sample = M1_1(sample, window)                                                   # M1_1 Window Multiply

'''
For Scalar Functions: Bias = Scalar, Windows = 0, Trig=False, Segments=1
For Segmented Functions, Bias and Windows need to both be the same length, Trig=False, and Segments=len(Bias)=len(Windows)
For Trig Functions, Windows is the amplitude of the trig wave

Standard Windows:
    Rectangular: Bias = [1], Windows = [0], Trig=False, Segments=1
    Triangular: Bias = [0,2], Windows = [2/N, -2/N], Trig=False, Segments=2
    Hanning: Bias = [0.5], Windows = [0.5], Trig=True, Segments=1
    Hamming: Bias = [0.54], Windows = [0.46], Trig=True, Segments=1
'''


#--------------------------------------------
# DFTmatrix Setup
#--------------------------------------------

DFTmatrix_file = 'DFTmatrix'

DFTmatrix = generate_DFTmatrix(DFTmatrix_file, N)                               #Generates the DFT Matrix

DFTraw = M1_2(sample, DFTmatrix)                                                # M1_2 DFT Multiply
DFTsums = S1(DFTraw)                                                            #Sums each frequency bin, this is a 1d array of the complex rectangular form of the DFT Terms


#--------------------------------------------
# DFT Output Setup
#--------------------------------------------
DFToutput_file = 'DFToutput'

DFToutput = Polar_Conversion(DFToutput_file, DFTsums)

'''
--------------------------------------------
PHASE 2 Conversion to Time Domain
--------------------------------------------

This section takes an array of amplitudes and phase angles in polar form
and produces a time domain output signal. 
'''

PhasorArray = DFTsums

#--------------------------------------------
# Mask Function Setup
#--------------------------------------------

# **WIP**

#--------------------------------------------
# IDFTmatrix Setup
#--------------------------------------------

IDFTmatrix_file = 'IDFTmatrix'

IDFTmatrix = generate_IDFTmatrix(IDFTmatrix_file, N)

#--------------------------------------------
# IDFT Output Setup
#--------------------------------------------

Signal_file = 'IDFToutput'

SignalArray = M2_2(PhasorArray, IDFTmatrix)
SignalOut = S2(Signal_file, SignalArray)

