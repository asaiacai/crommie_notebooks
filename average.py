#!/usr/bin/env python3
"""Zachary Goodwin
Python module averaging data
"""
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import matplotlib.cm as cm

#Defining the plotting stuff ... this was taken from the following websites.
#It just puts figures in a nice LaTeX format.

#http://bkanuka.com/articles/native-latex-plots/
#http://sbillaudelle.de/2015/02/23/seamlessly-embedding-matplotlib-output-into-latex.html

mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1)/2.0              # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.     "text.fontsize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1),       # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

#End of plotting stuff

Nx = 161#341
Ny = 121#681
Nz = 74
den = np.load('1.npy')

xc_XY = np.sum(den,axis=2)/Nz

plt.figure()
CF = plt.imshow(xc_XY, cmap='hot', interpolation='nearest')
cbar = plt.colorbar(CF)
cbar.ax.set_ylabel(r'')
plt.tight_layout()
plt.locator_params('x',tight=True, nbins =6)
plt.locator_params('y',tight=True, nbins =6)
#plt.savefig(folder+"DVM_real_"+I_ind+"_"+str(int(p))+str(int(q))+str(int(u))+str(int(v))+".pdf")
plt.show()
