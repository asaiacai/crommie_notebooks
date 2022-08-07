#!/usr/bin/env python3
"""Zachary Goodwin
Python module for plotting Bloch states
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
from matplotlib  import cm
import pandas as pd

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
    fig_height = fig_width*0.8#*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "sans-serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": ['calibri'],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.     "text.fontsize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1),       # default fig size of 0.9 textwidth
    "pgf.preamble": [
    #    r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
    #    r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

#End of plotting stuff

#####################
#Parameters for model
#####################
alpha = 10 #Binding energy in meV
epsilon = -150 #LUMO energy relative to Dirac point in meV
t = 3e3 #hopping parameter of graphene in meV
a = 1.42e-8#lattice constant of graphene in cm^-2
v_F = 1.5*t*a #Fermi velocity of graphene in cm^-2 multiplied by hbar
CV0 = (epsilon**2/(np.pi*v_F*v_F))*(1 - ((alpha+epsilon)/epsilon)**2) #onset density of molecules
N = 6.91E10*70 #maximum molecule density in cm^-2
sample = 1001 #how many points are included - need to rescale data such that this is the maximum value
rescale = 1e12 #convert meV/cm^2 to eV/nm^2

Ne = np.linspace(CV0,N+CV0,sample)
Ni = np.linspace(0,N,sample)
Delta_e = np.zeros((sample,sample))
##################################################################
#Calculating energy relative to equilibrium energy for a fixed N_e
##################################################################
for i in range(sample):
    Delta_e_0 = alpha*(Ne[i] - CV0) + epsilon*(Ne[i] - CV0) + 2*epsilon**3/(3*np.pi*v_F*v_F)*(1 - (1 - (np.pi*v_F*v_F)*(CV0)/epsilon**2)**(3/2))
    for j in range(sample):
        if ((np.pi*v_F*v_F)*(Ne[i]-Ni[j])/epsilon**2) < 1:
            Delta_e[i,j] = (alpha*Ni[j] + epsilon*Ni[j] + 2*epsilon**3/(3*np.pi*v_F*v_F)*(1 - (1 - (np.pi*v_F*v_F)*(Ne[i]-Ni[j])/epsilon**2)**(3/2)) - Delta_e_0)/rescale
        else:
            Delta_e[i,j] = (alpha*Ni[j] + epsilon*Ni[j] + 2*epsilon**3/(3*np.pi*v_F*v_F)*(1 - (-1 + (np.pi*v_F*v_F)*(Ne[i]-Ni[j])/epsilon**2)**(3/2)) - Delta_e_0)/rescale

################
#Plotting figure
################
fig, ax = plt.subplots(1,1)
Delta_e[np.where(Delta_e<=0.1)] = 0.1
CF = ax.imshow(Delta_e.T,cmap = cm.get_cmap('coolwarm', 1024) ,origin='lower', norm=mpl.colors.LogNorm(vmin=0.1, vmax=np.max(Delta_e)))

cbar = plt.colorbar(CF, shrink = 0.3, aspect = 10)
#cbar = plt.colorbar(cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=1e11, vmax=np.max(Delta_e)), cmap=cm.get_cmap('coolwarm', 1024)))
cbar.ax.set_ylabel(r'$\Delta E$ / GeV cm$^{-2}$')

TICK = [0,(sample-1)*(1E12/N),2*(sample-1)*(1E12/N),3*(sample-1)*(1E12/N),4*(sample-1)*(1E12/N)]
TICK_LAB = [0,1,2,3,4]
plt.locator_params('x',tight=True, nbins =7)
plt.locator_params('y',tight=True, nbins =7)
ax.set_xticks(TICK)
ax.set_yticks(TICK)
ax.set_xticklabels(TICK_LAB)
ax.set_yticklabels(TICK_LAB)
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
#plt.xlabel(r'$N_e^L$ - $\Delta N$ (10$^{12}$ cm$^{-2}$)', fontsize = 18)
plt.ylabel(r'Liquid phase density $N_l$  (10$^{12}$ cm$^{-2}$)', fontsize = 18)
plt.xlabel(r'$V_G$ - $V_0$ (V)', fontsize = 18)
N_12 = (float(N)/1.0E12)

def forward(x):
    return (N_12)*x/sample


def inverse(x):
    return sample/(N_12)*x

ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))
ax2.set_xlabel('$N_l^0$ (10$^{12}$ e cm$^{-2}$)', fontsize = 18)
ax2.set_xticks(TICK)
ax2.tick_params(axis='both', which='major', labelsize=14, width=2, length=5, direction='in', top = True)

ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=5, direction='in', right = True)
plt.rcParams['axes.linewidth'] = 2

exp_data = pd.read_csv('em26_856-897.csv')
x_data = exp_data['Unnamed: 0'][6:] + 10
y_data = exp_data['100 nm, avg'][6:]

x = np.linspace(0,sample-1,sample)
plt.plot(x,x,color='white',linewidth=0,linestyle='-.',label='Equilibrium density')

## overlay actual experimental data
p, V = np.polyfit(x_data, y_data, 1, cov=True)
fitted_C = 7.8e10
plt.plot(x_data/70*sample, y_data/N*sample, 'o', mfc='yellow', mec='none', markersize=8) #(fitted_C*x_data + p[1])/N*sample
x_gate = np.linspace(0,sample,8)
values=['{0:.0f}'.format(i) for i in np.linspace(0,70,8)]
ax.set_xticks(x_gate)
ax.set_xticklabels(values)
plt.xlim(0, sample) 
plt.ylim(0, sample)


N = sample
margin = 0.08
"""
plt.text(0.25*N,0.82*N,'Remove electrons',fontsize=20,color='black')
plt.text(0.12*N,0.35*N,'Chains grow',fontsize=20,color='black',rotation='vertical')
plt.arrow((0.8-margin/3)*N,0.8*N,(-0.6 + margin)*N,0,width=0.5,head_width = 5,color='black')
plt.arrow(0.2*N,(0.8-margin/3)*N,0,(-0.6 + margin)*N,width=0.5, head_width = 5, color='black')

plt.text(0.3*N,0.12*N,'Add electrons',fontsize=20,color='black')
plt.text(0.83*N,0.35*N,'Chains shrink',fontsize=20,color='black',rotation='vertical')
plt.arrow((0.2+margin/3)*N,0.2*N,(0.6 - margin)*N,0,width=0.5,head_width = 5,color='black')
plt.arrow(0.8*N,(0.2+margin/3)*N,0,(0.6 - margin)*N,width=0.5,head_width = 5,color='black')
"""
plt.tight_layout()
plt.savefig('PES.pdf')
plt.show()


"""
#Old version just with numbers, not densities

alpha = 10
epsilon = -150
A = 1*400

C = 750*400
V = 20
CV0 = -A*(epsilon**2)*(((alpha+epsilon)/epsilon)**2-1)

N = C*V
sample = 50
Ne = np.linspace(CV0,N+CV0,sample)
Ni = np.linspace(0,N,sample)
Delta_e = np.zeros((sample,sample))

for i in range(sample):
    Ni_0 = Ne[i] - CV0
    graphene_0 = - 2*A*epsilon**3*(1 - (1 - (CV0)/A/epsilon**2)**(3/2))/3
    Delta_e_0 = alpha*Ni_0 + epsilon*Ni_0 - graphene_0
    for j in range(sample):
        graphene = - 2*A*epsilon**3*(1 - (1 - (Ne[i]-Ni[j])/A/epsilon**2)**(3/2))/3
        Delta_e[i,j] = alpha*Ni[j] + epsilon*Ni[j] - graphene - Delta_e_0


"""