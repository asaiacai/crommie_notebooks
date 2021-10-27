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

from sxmreader import SXMReader
    
class MotionAnalyzer:
    """
    A class used to analyze motion of particles on a series of images.

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
    analyze():
        Performs batch particle tracking and linking of tracks, calculates diffusion constants
        and drifts.
    
    """
           
    def __init__(self, fileranges=None, voltages_temperatures=None, folder_name = None, heater = False, tracker_params = None):
        
        assert len(fileranges) == len(voltages_temperatures)
        self.heater = heater
        self.fileranges = fileranges
        self.voltages_temperatures = voltages_temperatures    
        self.SXM_PATH = [[folder_name + "/Image_{0:03}.sxm".format(i) for i in fileranges[j]] for j in range(len(fileranges))]
        self.SET_NAME = "{}-{}/".format(min([min(x) for x in fileranges]), max([max(x) for x in fileranges]))
        self.ANALYSIS_FOLDER = "./analysis/" + folder_name + "_" + self.SET_NAME
        self.MOVIE_FOLDER = self.ANALYSIS_FOLDER + "movies/"
        self.PARAMS_FILENAME = "params.yaml"
        if not os.path.exists(self.ANALYSIS_FOLDER):
            os.makedirs(self.ANALYSIS_FOLDER)
        if not os.path.exists(self.ANALYSIS_FOLDER):
            os.makedirs(self.MOVIE_FOLDER)
        self._set_search_params()
        #self.analyze_drift()
        #self.plot_average_drift()
    
    def analyze(self, plot_gif=False):
        self.drifts = []
        self.v_drift_mag = []
        self.D_constants = []
        self.D_constants2 = []
        self.msd_slope = []
        self.msd_intercept = []
        self.mu_hats = []
        self.ed = []
        self.em = []
        self.frames = []
        self.dataframes = []
        
        for i, path in enumerate(self.SXM_PATH):
            frames = SXMReader(path)
            self.frames.append(frames)
            self.NM_PER_PIXEL = frames.meters_per_pixel * 1e9 
            molecule_size, min_mass, max_mass, separation, min_size, max_ecc, adaptive_stop, search_range, _ = self.PARAMS[i]
            f = tp.batch(frames, molecule_size, minmass=min_mass, separation=separation)
            t = tp.link(f, search_range=search_range, adaptive_stop=adaptive_stop)
            t1 = t[((t['mass'] > min_mass) & (t['size'] > min_size) &
                 (t['ecc'] < max_ecc)) & (t['mass'] < max_mass)]
            t2 = tp.filter_stubs(t, 3)
            # Compare the number of particles in the unfiltered and filtered data.
            print('Before:', t['particle'].nunique())
            print('After:', t2['particle'].nunique())
            
            if plot_gif == True:
                moviename = "{}-{}".format(min(self.fileranges[i]), max(self.fileranges[i]))
                singlemoviefolder = self.MOVIE_FOLDER + moviename + "/"
                if not os.path.exists(singlemoviefolder):
                    os.makedirs(singlemoviefolder)
                mpl.rcParams.update({'font.size': 14, 'font.weight':'bold'})
                mpl.rc('image', origin='lower')
                mpl.rc('text',usetex =False)
                mpl.rc('text',color='orange')


                fns = []
                for j, frame in enumerate(frames):
                    fig= plt.figure(figsize=(5,5))
                    tp.plot_traj(t2[(t2['frame']<=j)], superimpose=frames[j], label=True)
                    fn = singlemoviefolder + "Image_{}.png".format(self.fileranges[i][j])
                    fig.savefig(fn)
                    fns.append(fn)
                    ax=plt.gca()                            # get the axis
                    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
                    ax.xaxis.tick_top()                     # and move the X-Axis      
                    ax.yaxis.set_ticks(np.arange(0, 16, 1)) # set y-ticks
                    ax.yaxis.tick_left()                    # remove right y-Ticks
                    plt.clf()
                mpl.rc('text',color='black')
                images = []
                for fn in fns:
                    images.append(imageio.imread(fn))
                imageio.mimsave(singlemoviefolder + moviename + '.gif', images, duration=0.5)
                self._cleanup_png(singlemoviefolder)

            # Compute drifts
            d = tp.compute_drift(t2)
            d.loc[0] = [0, 0]
            t3 = t2.copy()
            # Storing drifts
            self.drifts.append(d)
            
            # Method 1 of calculating D: variance of all displacements of Delta_t=1
            displacements = self._calculate_displacements(t3)
            self.D_constants.append((displacements.dx.var() + displacements.dy.var()) / 4) # r^2 = x^2 + y^2 = 2Dt + 2Dt
            self.mu_hats.append(np.mean(displacements[['dx', 'dy']], axis=0))

            
            # Method 2 of calculating D: linear fit to MSD
            em = tp.emsd(t3, frames.meters_per_pixel*1e9, self.DIFFUSION_TIME, max_lagtime=len(frames) ,detail=True)
            self.em.append(em)
            self.ed.append([em['<x>'],em['<y>']])
            result = linregress(em.index[:-8] * self.DIFFUSION_TIME, em['msd'][:-8])
            self.msd_slope.append(result.slope)
            self.msd_intercept.append(result.intercept)
            self.D_constants2.append(result.slope/4)

            # Store dataframe for future analysis
            self.dataframes.append(t3)
        self.v_drift_mag= np.linalg.norm(self.mu_hats, 2, axis=1)
            
    def _cleanup_png(self, singlemoviefolder):
        filelist = glob.glob(os.path.join(singlemoviefolder, "*.png"))
        for f in filelist:
            os.remove(f)
            
            
    def _calculate_displacements(self, t):
        displacements = pd.DataFrame()
        for j in range(t.frame.max() - 1):
                displacements = displacements.append(tp.relate_frames(t, j, j + 1) * self.NM_PER_PIXEL, ignore_index=True)
        return displacements

    
    
    def _set_search_params(self):
        with open('params.yaml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)       
        Params = namedtuple(
                    'Params', 
                    ['molecule_size', 
                     'min_mass',
                     'max_mass',
                     'separation',
                     'min_size',
                     'max_ecc',
                     'adaptive_stop',
                     'search_range',
                     'diffusion_time'])
        self.DIFFUSION_TIME = params['diffusion_time']
        self.PARAMS = [Params(**params) for i in range(len(self.voltages_temperatures))]
        copyfile(self.PARAMS_FILENAME, self.ANALYSIS_FOLDER + self.PARAMS_FILENAME)