# DSP-Tools
A set of tools for analyzing signals.

Included files:
```
  DSP Tools v02.py -- This is the python code for the DSP tools I have written. 
  DSP Tools Docs -- This is some basic documentation covering the different functions that make up DSP tools.
```
  Example Output Files:
    I created a test signal that looked kind of interesting using the wavegen on my Analog Discovery,
    and then ran my program to calculate the Discrete Fourier Transform (DFT) and generate plots of the
    Amplitudes and Phase Angles of each frequency bin. I then ran the DFT output through an Inverse Discrete
    Fourier Transform to calculate the time domain signal.
```
    Test Signal Sample.csv -- This is the sampled waveform from my Analog Discovery.
    
    DFT Raw Output.csv -- A csv file with calculated amplitude in column A, and phase angle in column B.
    DFT Amplitudes.svg -- A plot of the calculated amplitudes.
    DFT Phase Angles.svg -- A plot of the calculated phase angles.

    IDFT Raw Output.csv -- A csv file with the calculated sample values in column A.
    IDFT Signal Output.svg -- A plot of the IDFT output signal.
```
Dependencies:
```
  Numpy
  Datetime
  Matplotlib
```

Please note that my implementation of the DFT and IDFT are not optimized. This was meant as an educational
project that I created to learn some of the fundamentals of digital signal processing. The FFT and IFFT functions
built into Numpy would be significantly faster than the code I wrote.
    
