==============================================================================
libcuPaHOF: CUDA library to calculate PaHOF on a GPU in C++.
================
Fernando Cobo Aguilera and Manuel J. Marin-Jimenez


This software calculates the Pyramid of Accumulated Histograms of Optical Flow described in Marín et al. [1] 
--------------------------------------------------------------------------------

   Source-Code:   https://github.com/fcobo/libcupahof

--------------------------------------------------------------------------------
Contents of the package:
--------------------------------------------------------------------------------
- include - contains all the software header files
- src - contains all the software source files
- tests - contains a program to test the library
- makefile - used to compile the library, the documentation and the test program


--------------------------------------------------------------------------------
Requirements:
--------------------------------------------------------------------------------
This software has been tested on Windows 7 and Ubuntu 12.04 LTS (Precise Pangolin) with the following libraries:
- OpenCV - v2.4.8 (required)
- CUDA - v5.5 (required)


--------------------------------------------------------------------------------
Quick start:
--------------------------------------------------------------------------------
Let us assume that the root directory of libcupahof is named ‘rootdir’.

Open a terminal, and type in the command line the following instructions:
```
1) cd <rootdir>
2) mkdir build
3) cd build
4) cmake ..
5) make
6) make install   (You might need to do sudo if your are in an Unix-like system)
```
If everything went well, both the library and test programs should have been
created into <rootdir>/build subdirectories.


--------------------------------------------------------------------------------
Contact the authors:
--------------------------------------------------------------------------------
Fernando Cobo Aguilera (developer) - i92coagf@uco.es / fcoboaguilera@gmail.com
Manuel J. Marin-Jimenez (advisor) - mjmarin@uco.es


--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
[1] Marín-Jiménez, M. J.; Pérez de la Blanca, N.; Mendoza, M. A.; (2012): "Human
action recognition from simple feature pooling". Pattern Analysis and Applications
Journal, vol.17 no.1, On page(s): 17 - 36


--------------------------------------------------------------------------------
Version history:
--------------------------------------------------------------------------------

- v0.1: first release.
