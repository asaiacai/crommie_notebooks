#!/usr/bin/env python3
"""Zachary Goodwin
Python module formatting cube file
"""
import numpy as np 
import sys

Nx = 161#341
Ny = 121#681
Nz = 74
Nxyz = int(Nx*Ny*Nz)

f = open("1.dat", "r")
f1 = f.readlines()
fl = len(f1)

CUB = np.zeros((Nx,Ny,Nz))
cx = 0
cy = 0
cz = 0
c = 0
for i in range(fl):
    print(i)
    lin_f1 = f1[i]
 
    if c != 12:

        c+=1

        f1_x = np.zeros(6)
        f1_x[0] = float(lin_f1[1:13])
        f1_x[1] = float(lin_f1[14:26])
        f1_x[2] = float(lin_f1[27:39])
        f1_x[3] = float(lin_f1[40:52])
        f1_x[4] = float(lin_f1[53:65])
        f1_x[5] = float(lin_f1[66:78])

        for j in range(6):
            CUB[cx,cy,cz] = f1_x[j]

            cz += 1
            if cz == Nz:
                cz = 0
                cy += 1
                if cy == Ny:
                    cy = 0
                    cx += 1

    else:
        c = 0 

        f1_y = np.zeros(2)
        f1_y[0] = float(lin_f1[1:13])
        f1_y[1] = float(lin_f1[14:26])

        for j in range(2):
            CUB[cx,cy,cz] = f1_y[j]

            cz += 1
            if cz == Nz:
                cz = 0
                cy += 1
                if cy == Ny:
                    cy = 0
                    cx += 1

np.save('1',CUB)