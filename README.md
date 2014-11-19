============================================================
libcuPaHOF: CUDA library to calculate PaHOF on a GPU in C++.
============================================================
Fernando Cobo Aguilera and Manuel J. Marin-Jimenez


This software calculates the Pyramid of Accumulated Histograms of Optical Flow described in Marín-Jiménez et al. [1] 
--------------------------------------------------------------------------------

   Source-Code:   https://github.com/fcobo/libcupahof

--------------------------------------------------------------------------------
Contents of the package:
--------------------------------------------------------------------------------
- data - contains a pretrained SVM model for action recognition and test videos
- include - contains all the software header files
- src - contains all the software source files
- tests - contains a program to test the library


--------------------------------------------------------------------------------
Requirements:
--------------------------------------------------------------------------------
This software has been tested on Windows 7 and Ubuntu 12.04 LTS (Precise Pangolin) 
with the following libraries:
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
4) cmake ..       (If you want to generate the documentation, add the option -DINSTALL_DOC=yes)
5) make
6) make install   (You might need to do sudo if your are in an Unix-like system)
```
If everything went well, both the library and test programs should have been
created into <rootdir>/build subdirectories.

You can run the test program by executing the following command:
```
cd <rootdir>
cudahar --video data/person15_handwaving_d1_uncomp.avi 400 --hit_threshold 0 --gr_threshold 1 --classify data\train_svm.xml --gpu --fps 
```
A pretrained SVM on `KTH actions dataset' is used for classification. 


--------------------------------------------------------------------------------
Citation:
--------------------------------------------------------------------------------
If you use this library for your publications, please, consider citing the 
following publications:<br>
@article{marin2014paa,  
author = {Marin-Jimenez, M. J. and Perez de la Blanca, N. and Mendoza, M. A.},
 title  = {Human Action Recognition from simple feature pooling},
 year = {2014},
 journal = {Pattern Anal. Appl.},
 volume    = {17},
 number    = {1},
 pages     = {17--36}
}

@misc{libcupahof,  
author = {Cobo-Aguilera, Fernando and Marin-Jimenez, Manuel J.},
 title = {{LibCuPaHOF}: A CUDA library for computing {PaHOF} descriptors in {C++}},
 year = {2014},
 note =   {Software available at \url{https://github.com/fcobo/libcupahof}}
}


--------------------------------------------------------------------------------
Contact the authors:
--------------------------------------------------------------------------------
Fernando Cobo Aguilera (developer) - i92coagf@uco.es / fcoboaguilera@gmail.com<br>
Manuel J. Marin-Jimenez (advisor) - mjmarin@uco.es


--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
[1] Marín-Jiménez, M. J.; Pérez de la Blanca, N.; Mendoza, M. A.; (2014): "Human
action recognition from simple feature pooling". Pattern Analysis and Applications
Journal, vol.17 no.1, On page(s): 17 - 36


--------------------------------------------------------------------------------
Version history:
--------------------------------------------------------------------------------

- v0.1: first release.
