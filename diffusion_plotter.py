from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from collections import namedtuple
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import linregress
from scipy.optimize import curve_fit
import seaborn as sns

import numpy as np
import pandas as pd
import trackpy as tp

from pandas import DataFrame, Series  # for convenience
import os
from ntpath import basename
from shutil import copyfile
import yaml

from analyzer import MotionAnalyzer

class DiffusionPlotter(MotionAnalyzer):
    
    """
    A class used to plot diffusive motion of particles using results from a MotionAnalyzer.
    
    Attributes
    ----------
    fileranges : range []
        a list of filenumber ranges corresponding to each set of images, e.g. [range(801,815), range(815,829)]  
    voltages_temperatures : np.float32 [] 
        a list of voltages or temperatures corresponding to each set of images
    D_constants:
    
    drifts:
    em: DataFrame []
    Ensemble averaged Mean Square Displacement (EMSD)
    ed: [<x>,<y>]
    Ensemble averaged Displacement
    SXM_PATH 
    : [str []]
        a list of list of paths corresponding to the filenames of each image
    ANALYSIS_FOLDER : str
        the folder where the analysis results will be saved.
    

    Methods
    -------
    plot_drift_data()
    plot_diffusion()
    plot_msd()
    plot_ed()
    plot_v_over_D()
    
    """
    
    # Optionally, tweak styles.
    rc('animation', html='html5')
    mpl.rc('figure',  figsize=(5, 10))
    mpl.rc('image', cmap='gray')
    mpl.rc('image', origin='lower')
    mpl.rc('text',color='black')
    
    def __init__(self, ma: MotionAnalyzer):
        self.__dict__ = ma.__dict__.copy()
        
    def plot_msd(self):
        fig, ax = plt.subplots(figsize=(10,10))
        for i in range(len(self.em)):
            ax.plot(self.em[i].index * self.DIFFUSION_TIME, self.em[i]['msd'], 'o-', label= "{:.2f} $V_S$".format(self.voltages_temperatures[i]))
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [nm$^2$]',
            xlabel='lag time $t$')
            x = np.linspace(self.DIFFUSION_TIME,self.DIFFUSION_TIME * self.em[i].index,100)
            ax.plot(x, self.msd_slope[i]*x + self.msd_intercept[i], 'b--')
        plt.savefig(self.ANALYSIS_FOLDER + "msd.png")
    
    def plot_drift_vectors(self, plotrange = 20):
        mpl.rcParams.update({'font.size': 24, 'font.weight':'bold'})
        plt.figure(figsize=(10, 10))
        #colors = ['r', 'k', 'b', 'g', 'tab:orange', 'tab:purple', 'm']
        cmap = plt.cm.get_cmap("magma")
        colors = cmap(np.linspace(0,0.8,len(self.voltages_temperatures)))

        arrs = []
        j = 0
        for d in self.drifts:
            for i in range(1, len(d)):
                d0, d1 = d.loc[i - 1] * self.NM_PER_PIXEL, d.loc[i] * self.NM_PER_PIXEL
                plt.arrow(d0.x,d0.y,d1.x-d0.x, d1.y-d0.y, 
                shape='full', color=colors[j], length_includes_head=True, 
                zorder=0, head_length=0.5, head_width=0.5,linewidth=1)
            else:
                d0, d1 = d.loc[i - 1] * self.NM_PER_PIXEL, d.loc[i] * self.NM_PER_PIXEL
                arrs.append(plt.arrow(d0.x,d0.y,d1.x-d0.x, d1.y-d0.y, 
                shape='full', color=colors[j], length_includes_head=True, 
                zorder=0, head_length=0.5, head_width=0.5, label=str(self.voltages_temperatures[j])))
            j += 1
        new_labels, arrs = zip(*sorted(zip(self.voltages_temperatures, arrs)))
        new_labels=["{:.2f}".format(s) + ' V' for s in new_labels]
        plt.legend(arrs, new_labels, fontsize=16, loc='upper left')
        #plt.title("Ensemble Drift, " + SXM_PATH[0][0] + " to {}".format(SXM_PATH[-1][-1]))
        plt.xlabel("x (nm)",fontsize=24,fontweight='bold')
        plt.ylabel("y (nm)",fontsize=24,fontweight='bold')
        plt.xlim(-plotrange, plotrange)
        plt.ylim(-plotrange, plotrange)
        plt.savefig(self.ANALYSIS_FOLDER + "drift_vectors.png")
    
    def plot_drift_scalar(self,**kwargs):
        
        mag_displace = np.linalg.norm(self.mu_hats, 2, axis=1)
        new_labels, n_mag_displace, ord_D_constants = zip(*sorted(zip(self.voltages_temperatures, mag_displace, self.D_constants)))
        mpl.rcParams.update({'font.size' : 28, 'font.weight' : 'bold'})
        plt.figure(figsize=(10, 10))
        plt.plot(self.voltages_temperatures, mag_displace / self.DIFFUSION_TIME, '-o', markersize=18, linewidth=4)
        # plt.plot(xx, yy / 1.5)
        plt.ylabel('drift velocity (nm / s)')
        plt.xlabel('Voltage (V)')
        plt.savefig(self.ANALYSIS_FOLDER + "drift_scalar.png")
        
        plt.figure(figsize=(10, 10))
        mean_mu_hat = self._calculate_mean_axis(self.mu_hats)
        proj_mag_displace = np.array(self._project_to_mean_axis(self.mu_hats,mean_mu_hat))
        plt.plot(self.voltages_temperatures,  proj_mag_displace / self.DIFFUSION_TIME, '-o', markersize=18, linewidth=4)
        plt.ylabel('drift velocity (nm / s)')
        plt.xlabel('Voltage (V)')
        plt.savefig(self.ANALYSIS_FOLDER + "drift_scalar_projected.png")

    def _label_axes(self, ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    def plot_diffusion(self):
        font = {
        'weight' : 'bold',
        'size'   : 22}

        mpl.rc('font', **font)
        mpl.rc('text',usetex =True)
        fig, ax = plt.subplots(figsize=(10,10))
        tmpv, _sorted_D_constants = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.D_constants))))
        tmpv, _sorted_D_constants2 = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.D_constants2))))
        ax.plot(np.array(tmpv), _sorted_D_constants,'o-', label='variance')
        ax.plot(np.array(tmpv), _sorted_D_constants2,'o-', label='msd slope')
        ax.legend()
        
        if self.heater == True:
            self._label_axes(ax,'Temperature (K)','Diffusion constant ($nm^2$ / s)')
        else:
            self._label_axes(ax,'Voltage (V)','Diffusion constant ($nm^2$ / s)')
        plt.savefig(self.ANALYSIS_FOLDER + "D_constant_exp.png")

        fig, ax1 = plt.subplots(figsize=(10,10))
        sns.regplot(np.reciprocal(tmpv), np.log(_sorted_D_constants), 'o-', ci=None, ax=ax1)
        sns.regplot(np.reciprocal(tmpv), np.log(_sorted_D_constants2), 'o-', ci=None, ax=ax1)
        result = linregress(np.reciprocal(tmpv), np.log(_sorted_D_constants))
        result2 = linregress(np.reciprocal(tmpv), np.log(_sorted_D_constants2))

        
        if self.heater == True:
            self._label_axes(ax1,'1/T (1/K)','Log Diffusion constant ($nm^2$ / s)')
            ax1.annotate(r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(slope=result.slope,intercept = result.intercept),xy=(350,500), xycoords='figure pixels')
            ax1.annotate(r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(slope=result2.slope,intercept = result2.intercept),xy=(350,400), xycoords='figure pixels')
        else:
            self._label_axes(ax1,'1/V (1/V)','Log Diffusion constant ($nm^2$ / s)')
            ax1.annotate(r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(slope=result.slope,intercept = result.intercept),xy=(350,500), xycoords='figure pixels')
            ax1.annotate(r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(slope=result2.slope,intercept = result2.intercept),xy=(350,400), xycoords='figure pixels')
        
        plt.savefig(self.ANALYSIS_FOLDER + "logD_constant_lin.png")
    
    def _calculate_mean_axis(self, mu_hats):
        return sum(mu_hats)/len(mu_hats)
    
    def _project_to_mean_axis(self, mu_hats, mean_mu_hat):
        return [np.dot(v,mean_mu_hat) for v in mu_hats]
    
    def plot_drift_data(self):
        self.plot_drift_vectors()
        self.plot_drift_scalar()
          
    def make_gif(self):
        pass
    
    def plot_ed(self):
        fig, axs = plt.subplots(3)
        t = [i for i in range(1,len(m.ed[0][0])+1)]
        vx = []
        vy = []
        for i, volt in enumerate(self.voltages_temperatures):
            slope, intercept, _, _, _ = linregress(t[:-5],self.ed[i][0][:-5])
            #print("vx={:.2f}nm/s".format(slope))
            vx.append(slope)
            slope, intercept, _, _, _ = linregress(t[:-5],self.ed[i][1][:-5])
            #print("vy={:.2f}nm/s".format(slope))
            vy.append(slope)
        mpl.rcParams.update({'font.size': 24, 'font.weight':'bold'})


        axs[0].plot(self.voltages_temperatures,vx,'o-')
        axs[0].set_title('ensemble averaged vx')
        axs[1].plot(self.voltages_temperatures,vy,'o-')
        axs[1].set_title('ensemble averaged vy')
        axs[2].plot(self.voltages_temperatures,np.array(vx)**2 + np.array(vy)**2,'o-')
        axs[2].set_title('ensemble averaged msd')

        for i in range(3):
            axs[i].set_xlabel('voltage(V)')
            axs[i].set_ylabel('velocity (nm/s)')
            if i == 2:
                axs[i].set_ylabel('velocity (nm/$s^2$)')
        plt.savefig(self.ANALYSIS_FOLDER + "ensemble averaged v.png")
    
    def plot_v_over_D(self):
        
        def exponenial_func(x, a, b):
            return a * np.exp(-b / x )
        
        plt.figure(figsize=(7,5))
        popt, pcov = curve_fit(exponenial_func, self.voltages_temperatures, self.D_constants)

        xx = np.linspace(self.voltages_temperatures[0], self.voltages_temperatures[-1], 100)
        yy = exponenial_func(xx, *popt)
        plt.plot(xx, yy)
        plt.plot(self.voltages_temperatures, np.array(self.D_constants), 'o')
        plt.xlabel('$V_{SD} (V)$')
        plt.ylabel('$D (nm^2/s)$')

        plt.figure(figsize=(7,5))
        mag_displace = np.linalg.norm(self.mu_hats, 2, axis=1)
        popt1, pcov1 = curve_fit(exponenial_func, self.voltages_temperatures, mag_displace)
        yy1 = exponenial_func(xx, *popt1)
        plt.plot(xx, yy1)
        
        plt.plot(self.voltages_temperatures, mag_displace , 'o')
        plt.xlabel('$V_{SD} (V)$')
        plt.ylabel('$v_{drift} (nm/s)$')

        plt.figure(figsize=(7,5))
        yy2 = exponenial_func(xx, *popt1)/exponenial_func(xx, *popt)
        plt.plot(xx, yy2)
        plt.plot(self.voltages_temperatures, mag_displace/np.array(self.D_constants), 'o')
        plt.xlabel('$V_{SD} (V)$')
        plt.ylabel('$v_{drift}/D \ (1/nm)$') 