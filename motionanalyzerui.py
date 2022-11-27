from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os
import yaml
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import seaborn as sns
import ipywidgets as ipy
from IPython.display import display, clear_output, HTML
import pySPM as spm
import trackpy as tp
import deeptrack as dt
from tensorflow import keras
from tensorflow.keras import backend as K
from scipy.stats import linregress
from scipy.optimize import curve_fit


class ImageData:
    """
    A class used to retrieve/process the images and their properties.
    
    Attributes
    ----------
    foldername : str
        the name of the folder from where the images are being accessed, e.g. 'electromigration_00'
    setrange : range
        a range of filenumbers corresponding to each set of images, e.g. range(801,815) or range(815,829)
    filenames : str []
        a list of filenames corresponding to each SXM image
    zdata : array []
        a list of arrays corresponding to the raw z-channel pixels of each frame
    frames : array []
        a list of arrays corresponding to the processed and normalized (-1,1) zdata images
    pframes : array []
        a list of arrays specially processed and normalized (0,1) as input for the trained CNN model to generate masks
        
    
    Methods
    -------
    _frames_for_model():
        processes raw zdata into the necessary shape and normalization required for the trained CNN model
    
    """
    
    def __init__(self, foldername, setrange):
        self.foldername = foldername
        self.setrange = setrange
        self.filenames = [self.foldername+"/Image_{0:03}.sxm".format(i) for i in self.setrange]
        self.scans = [spm.SXM(filename) for filename in self.filenames]
        self.smallest = 9999999
        self.channel = "Z"
        self.zdata = []
        for j, s in enumerate(self.scans):
            scan = s.get_channel(self.channel).correct_lines().pixels
            with open(self.filenames[j], 'r', encoding='latin-1') as f:
                lines = f.readlines()
                scan_up = [
                    lines[i+1 % len(lines)] for i, x in enumerate(lines) if x == ":SCAN_DIR:\n"
                    ][0].strip() == "up"
                if not scan_up:
                    scan = scan[::-1]
            self.smallest = min(self.smallest, scan.shape[0])
            self.zdata.append(scan)
        self._len = len(self.zdata)
        self._dtype = np.float32
        self._frame_shape = (self.smallest, self.smallest)
        self.scan_size = self.scans[0].size
        self.nm_per_pixel = (self.scan_size['real']['x']/ self.smallest) * 1e9
        self.frames = dt.NormalizeMinMax(-1.0, 1.0).resolve(list(self.zdata))
        
        #get frames for the model
        def _frames_for_model():
            z_data = [i for i in self.zdata]
            inputframes = np.stack(z_data, axis=0)
            inputframes = np.expand_dims(inputframes[:], axis=-1)
            inputframes = dt.NormalizeMinMax(0, 1).resolve(list(inputframes))
            return inputframes
        self.pframes = _frames_for_model()


class AnalyzeUI(ipy.Box):
    
    """
    A class that contains all the necessary attributes, properties, and methods for the UI to function.
    
    Attributes
    ----------
    ROOT_FOLDER : str
        path to the current working directory
    
    """
    
    def __init__(self):
        super().__init__()
        self.ROOT_FOLDER = "./analysis/"
        self.usemodel = 'True'
        self.shift_i = 'True'
        self.nsize = []
        self.voltstemps = []
        self.setranges = []
        self.imsets = []
        self.locs = []
        self.links = []
        self.dlinks = []
        self.msize = []
        self.mass = []
        self.size = []
        self.maxecc = []
        self.sep = []
        self.tthresh = []
        self.pthresh = []
        self.srange = []
        self.memory = []
        self.astop = []
        self.filterstubs = []
        self.anchors = []
        self.drifts = []
        self.displacements1 = []
        self.D_constants1 = []
        self.mu_hats = []
        self.em = []
        self.ed = []
        self.msd_slope = []
        self.msd_intercept = []
        self.D_constants2 = []
        self.DIFFUSION_TIME = 1
        self.index = 0
        
        #define output widgets for plot        
        output1 = ipy.Output(
            layout=ipy.Layout(
                height='424px',
                width='424px',
                margin='0px 0px 0px 0px',
                padding='0px 0px 0px 0px',
                align_items='center',
                justify_content='center'
            ))
        
        output2 = ipy.Output(
            layout=ipy.Layout(
                height='269px',
                width='269px',
                margin='0px 0px 0px 0px',
                padding='0px 0px 0px 0px',
                justify_content='center'
            ))
        
        output3 = ipy.Output(
            layout=ipy.Layout(
                height='269px',
                width='269px',
                margin='0px 0px 0px 0px',
                padding='0px 0px 0px 0px',
                justify_content='center'
            ))
        
        output4 = []
        for i in range(6):
            output4.append(ipy.Output(
                layout=ipy.Layout(
                    height='100%',
                    width='100%',
                    margin='0px 0px 0px 0px',
                    padding='0px 0px 0px 0px',
                    justify_content='center',
                    align_items='center',
                )))
        
        #event handler for animated output        
        with output1:
            fig1, ax1 = plt.subplots(figsize=(6.0, 6.0), tight_layout=True)
            ln1, = ax1.plot([], [], lw=3)
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)
            plt.show()
        
        #definte initial settings subgroup
        submit_btn = ipy.Button(
            description='Submit',
            button_style='danger',
            disabled=True,
            layout=ipy.Layout(border='1px solid gray')
            )
        htmltxt = "<center><p style='font-size:12px'>{}</p>"
        folderslist = [f.name for f in os.scandir() if f.is_dir() and f.name.startswith('electromigration')]
        folders_tt = 'Select the electromigration folder'
        folders_wg = ipy.Dropdown(
            options=folderslist,
            value=None,
            style={'description_width':'initial'},
            layout=ipy.Layout(
                width='150px'
            ))        
        folders_box = ipy.VBox(
            [ipy.HTML(description=htmltxt.format('Folders:'), description_tooltip=folders_tt), folders_wg],
            layout=ipy.Layout(
                padding='0px 0px 5px 0px',
                justify_content='center',
                align_items='center'
            )
            )
        usemodel_tt = 'Use prediction masks from trained model for use with TrackPy.'
        usemodel_wg = ipy.Checkbox(
            value=False,
            indent=False,
            layout=ipy.Layout(
                width='55px',
                align_items='center',
                justify_content='center'
            ))
        usemodelwg_box = ipy.VBox(
            [ipy.HTML(
                description=htmltxt.format('Use Model:'),
                description_tooltip=usemodel_tt,
                layout=ipy.Layout(width='57px')
                ), usemodel_wg],
            layout=ipy.Layout(
                width='60px',
                padding='0px 0px 5px 0px',
                margin='0px 0px 0px 0px',
                justify_content='center',
                align_items='center',
                align_content='center',
            ))
        shifti_tt = 'Shifts the calculation of set ranges if needed.'
        shifti_wg = ipy.Checkbox(
            value=True,
            indent=False,
            layout=ipy.Layout(
                width='50px',
                align_items='center',
                justify_content='center'
            ))
        shiftiwg_box = ipy.VBox(
            [ipy.HTML(
                description=htmltxt.format('Shift i:'),
                description_tooltip=shifti_tt,
                layout=ipy.Layout(width='53px')
                ), shifti_wg],
            layout=ipy.Layout(
                width='55px',
                padding='0px 0px 5px 0px',
                margin='0px 0px 0px 0px',
                justify_content='center',
                align_items='center',
                align_content='center'
            ))
        initialwg_descriptions = [
            'Initial Value:',
            'Final Value:',
            'No. of Values:',
            'Starting Image:',
            'Batch Size:'
            ]
        initialwg_widget = [ipy.FloatText, ipy.FloatText, ipy.IntText, ipy.IntText, ipy.IntText]
        initialwg_default_values = [0.4, 0.54, 8, 395, 15]
        initialwg_list = []
        initialwg_group = [folders_box]
        for i, j, in enumerate(initialwg_descriptions):
            wg = initialwg_widget[i](
                value=initialwg_default_values[i],
                style={'description_width':'initial'},
                layout=ipy.Layout(width='60px')
                )
            box = ipy.VBox(
                [ipy.HTML(description=htmltxt.format(j), description_tooltip=initialwg_descriptions[i]), wg],
                layout=ipy.Layout(
                    padding='0px 0px 5px 0px',
                    justify_content='center',
                    align_items='center'
                ))
            initialwg_list.append(wg)
            initialwg_group.append(box)
        initialwg_group.append(usemodelwg_box)
        initialwg_group.append(shiftiwg_box)
        initialwg_group.append(submit_btn)
        
        #define tp.batch parameters subgroup
        paramwg_descriptions = [
            'Molecule Size: ',
            'Max ECC: ',
            'Separation: ',
            'TP Threshold: ',
            'Mask Threshold: '
            ]
        paramwg_options = [
            ['3', '5', '7', '9', '11', '13', '15'],
            ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5'],
            ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'],
            ['0.1', '0.06', '0.05', '0.025', '0.003'],
            ['0.91', '0.93', '0.95', '0.97', '0.99']
            ]
        paramwg_tts = [
            '''This may be a single number or a tuple giving the feature’s extent in each
            dimension, useful when the dimensions do not have equal resolution (e.g. confocal microscopy).''',
            'Filters out features with ECC greater than set threshold.',
            'Minimum separation between features. Default is diameter +1.',
            '''Clip bandpass result below this value. Thresholding is done on the already
            background-subtracted image.''',
            'Threshold on prediction mask provided by the trained model.'
        ]
        paramwg_default_values = ['3', '1.0', '2.0', '0.06', '0.99']
        paramwg_layouts = ['150px', '135px', '140px', '160px', '170px']
        paramwg_subgroup = []
        for i, j, in enumerate(paramwg_descriptions):
            wg = ipy.Dropdown(
                description=j,
                description_tooltip=paramwg_tts[i],
                options=paramwg_options[i],
                value=paramwg_default_values[i],
                style={'description_width':'initial'},
                layout=ipy.Layout(
                    width=paramwg_layouts[i],
                    justify_content='center',
                    align_items='center'
                ))
            paramwg_subgroup.append(wg)
        
        #define tp.link settings subgroup
        linkwg_descriptions = [
            'Search Range: ',
            'Memory: ',
            'Adaptive Stop: ',
            'Filter_Stubs: ',
            'Anchor?: '
            ]
        linkwg_options = [
            ['10', '20', '30', '40', '50', '60', '70', '80', '90'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'],
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            ['None']
            ]
        linkwg_tts = [
            'The maximum distance features can move between frames, optionally per dimension.',
            '''The maximum number of frames during which a feature can
            vanish,then reappear nearby, and be considered the same particle.''',
            '''If not None, when encountering an oversize subnet, retry
            by progressively reducing search_range until the subnet is solvable.''',
            'Filter out trajectories with fewer than selected points. They are often spurious.',
            '[Required] Confirm whether frames contain an anchor.'
        ]
        linkwg_default_values = ['30', '5', '2.0', '0', None]
        linkwg_layouts = ['150px', '120px', '160px', '150px', '130px']
        linkwg_subgroup = []
        for i, j in enumerate(linkwg_descriptions):
            wg = ipy.Dropdown(
                description=j,
                description_tooltip=linkwg_tts[i],
                options=linkwg_options[i],
                value=linkwg_default_values[i],
                style={'description_width':'initial'},
                layout=ipy.Layout(width=linkwg_layouts[i])
                )
            linkwg_subgroup.append(wg)
        anchor_box = ipy.Box(
            [linkwg_subgroup[4]],
            layout=ipy.Layout(
                height='50px',
                width='160px',
                padding='1px 0px 1px 0px',
                margin='1px 0px 1px 0px',
                justify_content='center',
                align_items='center',
                border='1px solid red'
            ))
        linkanchor_subgroup = [*linkwg_subgroup[:4], anchor_box]
            
        floatsliderwg_descriptions = ['Mass', 'Size']
        floatsliderwg_default_values = [[1.80, 5.00], [0.40, 1.00]]
        floatsliderwg_min_max = [[1.20, 7.00], [0.20, 3.00]]
        floatsliderwg_stepsize = [0.02, 0.02]
        floatsliderwg_readoutformat = ['.2f', '.2f']
        floatsliderwg_tts = [
            'Sets mass threshold filter for features.',
            'Sets size threshold filter for features.'
            ]
        floatsliderwg_subgroup = []
        for i, j in enumerate(floatsliderwg_descriptions):
            floatsliderwg_subgroup.append(ipy.FloatRangeSlider(
                description=j,
                description_tooltip=floatsliderwg_tts[i],
                value=floatsliderwg_default_values[i],
                min=floatsliderwg_min_max[i][0],
                max=floatsliderwg_min_max[i][1],
                step=floatsliderwg_stepsize[i],
                continuous_update=False,
                readout=True,
                readout_format=floatsliderwg_readoutformat[i],
                orientation='vertical',
                layout=ipy.Layout(
                    width='75px',
                    padding='0px 0px 0px 0px',
                    margin='0px 0px 0px 0px'
                    
                )))
        floatslider_box = ipy.VBox(
            floatsliderwg_subgroup,
            layout=ipy.Layout(
                height='539px',
                padding='0px 0px 0px 0px',
                margin='0px 0px 0px 0px',
                justify_content='space-around',
                align_items='center',
                ))        
        
        #vertical box for parameters subgroup
        param_box = ipy.VBox(
            paramwg_subgroup,
            layout=ipy.Layout(
                height='268px',
                width='180px',
                padding='5px 0px 5px 0px',
                margin='0px 0px 0px 0px',
                justify_content='space-around',
                align_items='center'
                ))
        
        #vertical box for link subgroup
        link_box = ipy.VBox(
            linkanchor_subgroup,
            layout=ipy.Layout(
                height='268px',
                width='180px',
                padding='5px 0px 5px 0px',
                margin='0px 0px 0px 0px',
                justify_content='space-around',
                align_items='center'
                ))
        
        paramlink_box = ipy.VBox(
            [param_box, link_box],
            layout=ipy.Layout(
                padding='0px 0px 0px 0px',
                margin='0px 0px 0px 0px'
                ))
        
        paramlinkslider_box = ipy.HBox(
            [paramlink_box, floatslider_box],
            layout=ipy.Layout(
                padding='0px 0px 0px 0px',
                margin='0px 0px 0px 0px',
                border='1px solid gray'
                ))
        
        next_btn = ipy.Button(
            description='Next',
            button_style='primary',
            disabled=False,
            layout=ipy.Layout(width='90px', border='1px solid gray')
            )
        back_btn = ipy.Button(
            description='Back',
            button_style='primary',
            disabled=True,
            layout=ipy.Layout(width='90px', border='1px solid gray')
            )
        navigation_btns = ipy.HBox(
            [back_btn, next_btn],
            layout=ipy.Layout(
                height='50px',
                width='300px',
                align_items='center',
                justify_content='center'
            ))
        
        checkbox_descriptions = ['Labels', 'Tracks', 'Heater?']
        checkbox_tts = [
            'Toggles display of particle labels from plot.',
            'Toggles display of particle motion tracks from plot.',
            'Declares whether a heater was used for this set.'
            ]
        checkboxwg_subgroup = []
        for j, i in enumerate(checkbox_descriptions):
            checkboxwg_subgroup.append(
                ipy.Checkbox(
                    value=True,
                    description=i,
                    description_tooltip=checkbox_tts[j],
                    indent=False,
                    layout=ipy.Layout(
                        width='100px',
                        justify_content='center'
                    )))
        checkboxwg_subgroup[2].value = False
        
        checkboxwg_box = ipy.HBox(
            checkboxwg_subgroup,
            layout=ipy.Layout(
                height='30px',
                width='300px',
                margin='0px 0px 10px 0px',
                align_items='center',
                justify_content='center',
                border='1px solid gray'
            ))
        
        mainoutput_box = ipy.VBox(
            [checkboxwg_box, output1, navigation_btns],
            layout=ipy.Layout(
                padding='0px 0px 0px 0px',
                margin='0px 0px 0px 0px',
                align_items='center',
                justify_content='space-around'
            ))
        
        suboutputs_box = ipy.VBox(
            [output2, output3],
            layout=ipy.Layout(
                height='539',
                border='1px solid gray'
            ))
        
        #horizontal box for link parameters group (subgroup & plot)
        subsetup_box = ipy.HBox(
            [paramlinkslider_box, mainoutput_box, suboutputs_box],
            layout=ipy.Layout(
                height='560px',
                width='100%',
                margin='0px 0px 0px 0px',
                align_items='center',
                justify_content='center',
                border='1px solid gray',
                visibility='hidden'
            ))
        
        #horizontal box for intital settings group
        settings_box = ipy.HBox(
            initialwg_group,
            layout=ipy.Layout(
                height='75px',
                border='1px solid gray',
                padding='1px 1px 1px 1px',
                justify_content='space-around',
                align_items='center'
            ))
        
        #define secondary control header group
        analyze_btn = ipy.Button(
            description='Analyze',
            button_style='danger',
            disabled=True,
            layout=ipy.Layout(border='1px solid gray')
            )
        infotext_wg = ipy.HTML(
            value="<p style='font-size:24px'><b>Error</b></p>",
            layout=ipy.Layout(
                width = '625px',
            ))
        
        headerwg_group = ipy.HBox(
            [infotext_wg, analyze_btn],
            layout=ipy.Layout(
                height='75px',
                border='1px solid gray',
                padding='1px 1px 1px 1px',
                justify_content='space-around',
                align_items='center'
            ))
        
        initialinfo_text1 = ipy.HTML(
            value="<center><p style='font-size:18px'><b>Current Voltages/Temperatures:</b></p>",
            layout=ipy.Layout(
                width='98%',
                padding='15px 10px 15px 10px',
                margin='15px 10px 15px 10px'
            ))
        initialinfo_text2 = ipy.HTML(
            value="<center><p style='font-size:18px'><b>Current Set Ranges:</b></p>",
            layout=ipy.Layout(
                width='98%',
                padding='15px 10px 15px 10px',
                margin='15px 10px 15px 10px'
            ))
        initialinfo_box = ipy.VBox(
            [initialinfo_text1, initialinfo_text2],
            layout=ipy.Layout(
                width='99%',
                padding='10px 10px 10px 10px',
                margin='10px 10px 10px 10px',
                justify_content='flex-start',
                align_items='center'
            ))
        
        #define overall UI layout and sizing (do not change these)
        def ui_box(b):
            if b==0:
                setup_box = ipy.Box(
                    [settings_box, initialinfo_box],
                    layout=ipy.Layout(
                        display='flex',
                        flex_flow='column',
                        height='615px',
                        width='100%',
                        justify_content='flex-start',
                        margin='0px 0px 0px 0px',
                        border='1px solid gray'
                        ))
            else:
                setup_box = ipy.Box(
                    [headerwg_group, subsetup_box],
                    layout=ipy.Layout(
                        display='flex',
                        flex_flow='column',
                        height='615px',
                        width='100%',
                        justify_content='flex-start',
                        margin='0px 0px 0px 0px',
                        border='1px solid gray'
                        ))
            plot_box = [setup_box]
            for i in range(len(output4)):
                plot_box.append(ipy.Box(
                    [output4[i]],
                    layout=ipy.Layout(
                        display='flex',
                        flex_flow='column',
                        height='615px',
                        width='100%',
                        margin='0px 0px 0px 0px',
                        border='1px solid gray'
                        )))
            plotnames = ['Drift Vectors', 'Drift Scalars', 'Diffusion', 'MSD', 'ED', 'V Over D']
            tabs = ipy.Tab(
                children=plot_box,
                layout=ipy.Layout(
                    height='675px',
                    width='100%',
                    margin='0px 0px 0px 0px'
                    ))
            tabs.set_title(0, 'Setup')            
            for i in range(6):
                tabs.set_title(i+1, plotnames[i])
            
            header_txt = """
                <center><p style='font-size:15px'><font color='#FDB515'><b>MotionAnalyzeUI Mk.5</b></p>
                """
            header_bar = ipy.HTML(
                value=header_txt,
                layout=ipy.Layout(
                    height='25px',
                    width='100%',
                    margin='0px 0px 0px 0px'
                ))
            header_bar.add_class('box_bg')
            return ipy.VBox(
                [header_bar, tabs],
                layout=ipy.Layout(
                    height='700px',
                    width='100%',
                    border='1px solid gray'
                ))
        
        def make_dir():
            foldername = folders_wg.value+'_{}-{}'.format(self.setranges[0][0], self.setranges[-1][1])
            moviepath = os.path.join(self.ROOT_FOLDER, foldername, 'movies')
            os.makedirs(moviepath, exist_ok=True)
            
        def save_toyaml():
            foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
            yaml_keys = [
                'molecule_size',
                'mass',
                'size',
                'max_ecc',
                'separation',
                'trackpy_threshold',
                'model_threshold',
                'search_range',
                'memory',
                'adaptive_stop',
                'filter_stubs'
                ]
            yaml_vars = [
                self.msize,
                self.mass,
                self.size,
                self.maxecc,
                self.sep,
                self.tthresh,
                self.pthresh,
                self.srange,
                self.memory,
                self.astop,
                self.filterstubs
                ]
            yaml_dict = dict(zip(yaml_keys, yaml_vars))
            with open(self.ROOT_FOLDER+foldername+'params.yaml', 'w') as file:
                documents = yaml.dump(yaml_dict, file)
                
        def loadingscreen(ls):
            for i in [output1, output2, output3]:
                with i:
                    i.clear_output(True)
                    display(HTML('<img src="loading.gif">'))
        
        def update0(i):
            for txt in ax1.texts:
                txt.set_visible(False)
            tp.plot_traj(
                self.links[self.index][self.links[self.index]['frame']<=i],
                superimpose=None,
                label=True,
                ax=ax1
                )
            tp.annotate(
                self.locs[self.index][self.locs[self.index]['frame']==i],
                self.imsets[self.index].frames[i],
                ax=ax1
                )
            ax1.set_prop_cycle(color=['g', 'r', 'c', 'm', 'y', 'b'])
            for txt in ax1.texts:
                txt.set_fontsize(24)
                txt.set_color('darkorange')
            for line in ax1.lines:
                line.set_markersize(3)
            for line in ax1.lines[:-1]:
                line.set_marker(None)
            return ln1,
        
        def update1(i):
            for txt in ax1.texts:
                txt.set_visible(False)
            if checkboxwg_subgroup[0].value == True:
                if checkboxwg_subgroup[1].value == True:
                    tp.plot_traj(
                        self.links[self.index][self.links[self.index]['frame']<=i],
                        superimpose=None,
                        label=True,
                        ax=ax1
                        )
                elif checkboxwg_subgroup[1].value == False:
                    tp.plot_traj(
                        self.links[self.index][self.links[self.index]['frame']==i],
                        superimpose=None,
                        label=True,
                        ax=ax1
                        )
            elif checkboxwg_subgroup[1].value == True:
                tp.plot_traj(
                    self.links[self.index][self.links[self.index]['frame']<=i],
                    superimpose=None,
                    label=False,
                    ax=ax1
                    )
            elif checkboxwg_subgroup[0].value and checkboxwg_subgroup[1].value == False:
                pass
            tp.annotate(
                self.locs[self.index][self.locs[self.index]['frame']==i],
                self.imsets[self.index].frames[i],
                ax=ax1
                )
            ax1.set_prop_cycle(color=['g', 'r', 'c', 'm', 'y', 'b'])
            for txt in ax1.texts:
                txt.set_fontsize(24)
                txt.set_color('darkorange')
            for line in ax1.lines:
                line.set_markersize(3)
            for line in ax1.lines[:-1]:
                line.set_marker(None)
            return ln1,
        
        def save_output():
            loadingscreen('all')
            with output1:
                ax1.clear()
                ani = FuncAnimation(
                    fig1,
                    update0,
                    frames=range(len(self.imsets[self.index].frames)),
                    blit='True',
                    interval=300)
                foldername = folders_wg.value+'_{}-{}/movies'.format(
                    self.setranges[0][0],
                    self.setranges[-1][1]
                    )
                filename = '{}-{}.gif'.format(
                    self.setranges[self.index][0],
                    self.setranges[self.index][1]
                    )
                ani.save(
                    filename,
                    writer=PillowWriter(fps=3)
                    );
                shutil.move(filename, self.ROOT_FOLDER+foldername)
        
        def update_output1(i):
            with output1:
                ax1.clear()
                output1.clear_output(True)
                ani = FuncAnimation(
                    fig1,
                    update1,
                    frames=range(len(self.imsets[self.index].frames)),
                    blit='True',
                    interval=300)
                display(HTML(ani.to_html5_video()))
                
        def update_output2(i):
            with output2:
                fig2, ax2 = plt.subplots(figsize=(3.6, 3.6), tight_layout=True)
                ln2, = ax2.plot([], [], lw=3)
                output2.clear_output(True)
                tp.mass_size(self.links[self.index].groupby('particle').mean(), ax=ax2);
                plt.show(fig2)
                
        def update_output3(i):
            with output3:
                fig3, ax3 = plt.subplots(figsize=(3.6, 3.6), tight_layout=True)
                ln3, = ax3.plot([], [], lw=3)
                output3.clear_output(True)
                tp.mass_ecc(self.links[self.index].groupby('particle').mean(), ax=ax3);
                plt.show(fig3)
        
        def get_currentranges():
            self.voltstemps = np.linspace(
                float(initialwg_list[0].value),
                float(initialwg_list[1].value),
                int(self.nsize),
                dtype=np.float32)
            if self.shift_i == 'False':
                self.setranges = [range(
                    initialwg_list[3].value+1+initialwg_list[4].value*i,
                    initialwg_list[3].value+initialwg_list[4].value*(i+1)
                    ) for i in range(self.nsize)]
            else:
                self.setranges = [range(
                    initialwg_list[3].value+i+initialwg_list[4].value*i,
                    initialwg_list[3].value+i+initialwg_list[4].value*(i+1)
                    ) for i in range(self.nsize)]
                    
        def get_initialsettings():
            batch_parameters = [self.msize, self.maxecc, self.sep, self.tthresh, self.pthresh]
            link_parameters = [self.srange, self.memory, self.astop, self.filterstubs, self.anchors]
            slider_parameters = [self.mass, self.size]
            sets = [self.imsets, self.locs, self.links, self.dlinks, self.drifts]
            (i.clear() for i in sets)
            (j.clear() for j in batch_parameters)
            (k.clear() for k in link_parameters)
            batch_vals = [i.value for i in paramwg_subgroup]
            link_vals = [i.value for i in linkwg_subgroup]
            slider_vals = [i.value for i in floatsliderwg_subgroup]
            if self.usemodel == 'False':
                batch_vals = ['9', '1.5', '2.0', '0.06', '0.99']
                link_vals = ['50', '5', '2.0', '0', None]
                slider_vals = [(2.00,8.00),(0.40, 2.00)]
            for i in range(self.nsize):
                for l in sets:
                    l.append(None)
                for j, k in enumerate(batch_parameters):
                    k.append(batch_vals[j])
                for m, n in enumerate(link_parameters):
                    n.append(link_vals[m])
                for p, q in enumerate(slider_parameters):
                    q.append(slider_vals[p])
            
        def get_imagedata():
            imsets = []
            for i in range(self.nsize):
                imsets.append(ImageData(folders_wg.value, self.setranges[i]))
            self.imsets = imsets

        def get_locations():
            tp.quiet()
            if self.usemodel == 'False':
                self.locs[self.index] = tp.batch(
                    self.imsets[self.index].frames,
                    int(self.msize[self.index]),
                    minmass=float(self.mass[self.index][0]),
                    separation=float(self.sep[self.index]),
                    characterize=True,
                    threshold=float(self.tthresh[self.index])
                    )
            else:
                predictions = self.model.predict(
                    np.array(self.imsets[self.index].pframes),
                    batch_size=1
                    )
                masks = []
                for i in range(len(predictions[:,:,:,0])):
                    prediction = predictions[i,:,:,0] > float(self.pthresh[self.index])
                    masks.append(np.squeeze(prediction))
                self.locs[self.index] = tp.batch(
                    masks,
                    int(self.msize[self.index]),
                    minmass=float(self.mass[self.index][0]),
                    separation=float(self.sep[self.index]),
                    characterize=True,
                    threshold=float(self.tthresh[self.index])
                    )
    
        def get_links():
            link0 = tp.link(
                self.locs[self.index],
                int(self.srange[self.index]),
                memory=int(self.memory[self.index]),
                adaptive_stop=float(self.astop[self.index]),
                adaptive_step=0.95
                )
            link1 = link0[((link0['mass'] > float(self.mass[self.index][0])) &
                           (link0['size'] > float(self.size[self.index][0])) & 
                           (link0['size'] < float(self.size[self.index][1])) &
                           (link0['ecc'] < float(self.maxecc[self.index]))) &
                           (link0['mass'] < float(self.mass[self.index][1]))]
            link2 = tp.filter_stubs(link1, int(self.filterstubs[self.index]))
            self.links[self.index] = link2
            
        def saveparams():
            batch_parameters = [self.msize, self.maxecc, self.sep, self.tthresh, self.pthresh]
            link_parameters = [self.srange, self.memory, self.astop, self.filterstubs, self.anchors]
            slider_parameters = [self.mass, self.size]
            for j, k in enumerate(batch_parameters):
                k[self.index] = paramwg_subgroup[j].value
            for m, n in enumerate(link_parameters):
                n[self.index] = linkwg_subgroup[m].value
            for p, q in enumerate(slider_parameters):
                q[self.index] = floatsliderwg_subgroup[p].value
                
                
        def loadparams():
            batch_parameters = [self.msize, self.maxecc, self.sep, self.tthresh, self.pthresh]
            link_parameters = [self.srange, self.memory, self.astop, self.filterstubs, self.anchors]
            slider_parameters = [self.mass, self.size]
            for j, k in enumerate(batch_parameters):
                paramwg_subgroup[j].value = k[self.index]
            for m, n in enumerate(link_parameters):
                linkwg_subgroup[m].value = n[self.index]
            for p, q in enumerate(slider_parameters):
                floatsliderwg_subgroup[p].value = q[self.index]

        def calculate_displacements(MPP, t):
            displacements = pd.DataFrame()
            for j in range(t.frame.max() - 1):
                displacements = displacements.append(
                    tp.relate_frames(t, j, j + 1) * MPP,
                    ignore_index=True
                    )
            return displacements
        
        def label_axes(ax, xlabel, ylabel, fontsize):
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            
        def plot_drift_vectors(MPP, plotrange = 20):
            with output4[0]:
                foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
                fig4, ax4 = plt.subplots(figsize=(8, 8), tight_layout=True)
                output4[0].clear_output(True)
                
                cmap = plt.cm.get_cmap("magma")
                colors = cmap(np.linspace(0, 0.8, self.nsize))
                arrs = []
                j = 0
                for d in self.drifts:
                    for i in range(1, len(d)):
                        d0, d1 = d.loc[i - 1] * MPP, d.loc[i] * MPP
                        plt.arrow(d0.x,d0.y,d1.x-d0.x, d1.y-d0.y, 
                        shape='full', color=colors[j], length_includes_head=True, 
                        zorder=0, head_length=0.5, head_width=0.5, linewidth=1)
                    else:
                        d0, d1 = d.loc[i - 1] * MPP, d.loc[i] * MPP
                        arrs.append(plt.arrow(d0.x,d0.y,d1.x-d0.x, d1.y-d0.y, 
                        shape='full', color=colors[j], length_includes_head=True, 
                        zorder=0, head_length=0.5, head_width=0.5, label=str(self.voltstemps[j])))
                    j += 1
                new_labels, arrs = zip(*sorted(zip(self.voltstemps, arrs)))
                new_labels=["{:.2f}".format(s) + ' V' for s in new_labels]
                plt.legend(arrs, new_labels, fontsize=16, loc='upper left')
                label_axes(ax4, "x (nm)", "y (nm)", 16)
                plt.xlim(-plotrange, plotrange)
                plt.ylim(-plotrange, plotrange)
                plt.savefig(self.ROOT_FOLDER+foldername+'drift_vectors.png')
                plt.show(fig4)
        
        def calculate_mean_axis(mu_hats):
            return sum(mu_hats)/len(mu_hats)
    
        def project_to_mean_axis(mu_hats, mean_mu_hat):
            return [np.dot(v, mean_mu_hat) for v in mu_hats]

        def plot_drift_scalar(**kwargs):
            with output4[1]:
                foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
                fig5, ax5 = plt.subplots(1, 2, figsize=(13, 6))
                output4[1].clear_output(True)
                
                mag_displace = np.linalg.norm(self.mu_hats, 2, axis=1)
                new_labels, n_mag_displace, ord_D_constants = zip(*sorted(zip(
                    self.voltstemps,
                    mag_displace,
                    self.D_constants1
                    )))
                ax5[0].plot(
                    self.voltstemps,
                    mag_displace/self.DIFFUSION_TIME,
                    '-o',
                    markersize=18,
                    linewidth=4
                    )
                label_axes(ax5[0], 'Voltage (V)', 'drift velocity (nm / s)', 16)

                mean_mu_hat = calculate_mean_axis(self.mu_hats)
                proj_mag_displace = np.array(project_to_mean_axis(self.mu_hats, mean_mu_hat))
                ax5[1].plot(
                    self.voltstemps,
                    proj_mag_displace/self.DIFFUSION_TIME,
                    '-o',
                    markersize=18,
                    linewidth=4
                    )
                label_axes(ax5[1], 'Voltage (V)', 'drift velocity (nm / s)', 16)
                
                fig5.tight_layout(pad=2.0)
                for i, j in zip([ax5[0],ax5[1]],['drift_scalar.png','drift_scalar_projected.png']):
                    extent = i.get_tightbbox(fig5.canvas.renderer).transformed(fig5.dpi_scale_trans.inverted())
                    plt.savefig(
                        self.ROOT_FOLDER+foldername+j,
                        bbox_inches=extent.expanded(1.05, 1.1)
                        )
                plt.show(fig5)

        def plot_diffusion(heater):
            with output4[2]:
                foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
                fig6, ax6 = plt.subplots(1, 2, figsize=(13, 6))
                output4[2].clear_output(True)
                
                tmpv, sorted_D_constants1 = (list(t) for t in zip(*sorted(zip(
                    self.voltstemps,
                    self.D_constants1
                    ))))
                tmpv, sorted_D_constants2 = (list(t) for t in zip(*sorted(zip(
                    self.voltstemps,
                    self.D_constants2
                    ))))
                ax6[0].plot(np.array(tmpv), sorted_D_constants1,'o-', label='variance')
                ax6[0].plot(np.array(tmpv), sorted_D_constants2,'o-', label='msd slope')
                ax6[0].legend(fontsize=16)

                if heater == 'True':
                    label_axes(ax6[0], 'Temperature (K)', 'Diffusion constant ($nm^2$ / s)', 16)
                else:
                    label_axes(ax6[0], 'Voltage (V)', 'Diffusion constant ($nm^2$ / s)', 16)

                sns.regplot(np.reciprocal(tmpv), np.log(sorted_D_constants1), 'o-', ci=None, ax=ax6[1])
                sns.regplot(np.reciprocal(tmpv), np.log(sorted_D_constants2), 'o-', ci=None, ax=ax6[1])
                result = linregress(np.reciprocal(tmpv), np.log(sorted_D_constants1))
                result2 = linregress(np.reciprocal(tmpv), np.log(sorted_D_constants2))

                if heater == 'True':
                    label_axes(ax6[1], '1/T (1/K)', 'Log Diffusion constant ($nm^2$ / s)', 16)
                    ax6[1].annotate(
                        r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(
                            slope=result.slope,
                            intercept = result.intercept
                            ),
                        fontsize=16,
                        xy=(0.45,0.85),
                        xycoords='axes fraction'
                        )
                    ax6[1].annotate(
                        r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(
                            slope=result2.slope,
                            intercept = result2.intercept
                            ),
                        fontsize=16,
                        xy=(0.45,0.7),
                        xycoords='axes fraction'
                    )
                else:
                    label_axes(ax6[1], '1/V (1/V)', 'Log Diffusion constant ($nm^2$ / s)', 16)
                    ax6[1].annotate(
                        r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(
                            slope=result.slope,
                            intercept = result.intercept
                            ),
                        fontsize=16,
                        xy=(0.45,0.85),
                        xycoords='axes fraction'
                        )
                    ax6[1].annotate(
                        r'ln(D)= {slope:.2f} $\frac{{1}}{{T}}$+ {intercept:.2f}'.format(
                            slope=result2.slope,
                            intercept = result2.intercept
                            ),
                        fontsize=16,
                        xy=(0.45,0.7),
                        xycoords='axes fraction'
                        )
                
                fig6.tight_layout(pad=2.0)
                for i, j in zip([ax6[0],ax6[1]],['D_constant_exp.png','logD_constant_lin.png']):
                    extent = i.get_tightbbox(fig6.canvas.renderer).transformed(fig6.dpi_scale_trans.inverted())
                    plt.savefig(
                        self.ROOT_FOLDER+foldername+j,
                        bbox_inches=extent.expanded(1.05, 1.1)
                        )
                plt.show(fig6)

        def plot_msd():
            with output4[3]:
                foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
                fig7, ax7 = plt.subplots(figsize=(8, 8), tight_layout=True)
                output4[3].clear_output(True)
                for i in range(len(self.em)):
                    ax7.plot(
                        self.em[i].index * self.DIFFUSION_TIME, self.em[i]['msd'],
                        'o-',
                        label='{:.2f} '.format(self.voltstemps[i])+r'$\mathrm{V_S}$'
                        )
                    ax7.legend(fontsize=16)
                    ax7.set_xscale('log')
                    ax7.set_yscale('log')
                    label_axes(ax7, r'lag time $t$', r'$\langle \Delta r^2 \rangle$ [nm$^2$]', 16)
                    x = np.linspace(self.DIFFUSION_TIME, self.DIFFUSION_TIME * self.em[i].index, 100)
                    ax7.plot(x, self.msd_slope[i]*x + self.msd_intercept[i], 'b--')
                plt.savefig(self.ROOT_FOLDER+foldername+'msd.png')
                plt.show(fig7)

        def plot_ed():
            with output4[4]:
                foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
                output4[4].clear_output(True)
                fig8 = plt.figure(figsize=(12,8), constrained_layout=True)
                gs1 = mpl.gridspec.GridSpec(2,8, figure=fig8, wspace=0.8)
                axs1 = plt.subplot(gs1[0, :4])
                axs2 = plt.subplot(gs1[0, 4:])
                axs3 = plt.subplot(gs1[1, 1:5])
                
                t = [i for i in range(1,len(self.ed[0][0])+1)]
                vx = []
                vy = []
                for i, volt in enumerate(self.voltstemps):
                    slope, intercept, _, _, _ = linregress(t[:-5], self.ed[i][0][:-5])
                    vx.append(slope)
                    slope, intercept, _, _, _ = linregress(t[:-5], self.ed[i][1][:-5])
                    vy.append(slope)

                axs1.plot(self.voltstemps, vx, 'o-')
                axs1.set_title('ensemble averaged vx', fontsize=16)
                axs2.plot(self.voltstemps, vy, 'o-')
                axs2.set_title('ensemble averaged vy', fontsize=16)
                axs3.plot(self.voltstemps, np.array(vx)**2 + np.array(vy)**2, 'o-')
                axs3.set_title('ensemble averaged msd', fontsize=16)

                for i, j in enumerate([axs1, axs2, axs3]):
                    label_axes(j,'voltage(V)','velocity (nm/s)', 16)
                    if i == 2:
                        label_axes(j,'voltage(V)','velocity (nm/$s^2$)',16)
                
                plt.savefig(self.ROOT_FOLDER+foldername+'ensemble averaged v.png')
                plt.show(fig8)

        def plot_v_over_D():
        
            def exponenial_func(x, a, b):
                return a * np.exp(-b / x )
            
            with output4[5]:
                foldername = folders_wg.value+'_{}-{}/'.format(self.setranges[0][0], self.setranges[-1][1])
                output4[5].clear_output(True)
                fig9 = plt.figure(figsize=(12,8), constrained_layout=True)
                gs2 = mpl.gridspec.GridSpec(2,8, figure=fig9, wspace=1.0)
                axs4 = plt.subplot(gs2[0, :4])
                axs5 = plt.subplot(gs2[0, 4:])
                axs6 = plt.subplot(gs2[1, 1:5])
                popt, pcov = curve_fit(exponenial_func, self.voltstemps, self.D_constants1)

                xx = np.linspace(self.voltstemps[0], self.voltstemps[-1], 100)
                yy = exponenial_func(xx, *popt)
                axs4.plot(xx, yy)
                axs4.plot(self.voltstemps, np.array(self.D_constants1), 'o')
                label_axes(axs4, '$V_{SD} (V)$', '$D (nm^2/s)$', 16)

                mag_displace = np.linalg.norm(self.mu_hats, 2, axis=1)
                popt1, pcov1 = curve_fit(exponenial_func, self.voltstemps, mag_displace)
                yy1 = exponenial_func(xx, *popt1)
                axs5.plot(xx, yy1)
                axs5.plot(self.voltstemps, mag_displace , 'o')
                label_axes(axs5, '$V_{SD} (V)$', '$v_{drift} (nm/s)$', 16)

                yy2 = exponenial_func(xx, *popt1)/exponenial_func(xx, *popt)
                axs6.plot(xx, yy2)
                axs6.plot(self.voltstemps, mag_displace/np.array(self.D_constants1), 'o')
                label_axes(axs6, '$V_{SD} (V)$', '$v_{drift}/D \ (1/nm)$', 16)
                plt.savefig(self.ROOT_FOLDER+foldername+'v_over_D.png')
                plt.show(fig9)
                
        def changeval(change):
            self.nsize = initialwg_list[2].value
            get_currentranges()
            infotxt = "<center><p style='font-size:18px'><font color='#003262'><b>{}</b> {}</p>"
            initialinfo_text1.value = infotxt.format(
                'Current Voltages/Temperatures: ',
                str([round(float(i), 3) for i in self.voltstemps])
                )
            initialinfo_text2.value = infotxt.format('Current Set Ranges: ', str(self.setranges))
                
        def changeslider(change):
            x = [self.mass, self.size]
            for i, j in enumerate(x):
                if change['owner'].description == floatsliderwg_subgroup[i]:
                    j[self.index] = change.new
            loadingscreen('all')
            get_locations()
            get_links()
            update_output1(self.index)
            update_output2(self.index)
            update_output3(self.index)                
                
        def changelinks(change):
            x = [self.srange, self.memory, self.astop, self.filterstubs]
            for i, j in enumerate(x):
                if change['owner'].description == linkwg_descriptions[i]:
                    j[self.index] = change.new
            loadingscreen('all')
            get_locations()
            get_links()
            update_output1(self.index)
            update_output2(self.index)
            update_output3(self.index)
        
        def changeparams(change):
            x = [self.msize, self.maxecc, self.sep, self.tthresh, self.pthresh]
            for i, j in enumerate(x):
                if change['owner'].description == paramwg_descriptions[i]:
                    j[self.index] = change.new
            loadingscreen('all')
            get_locations()
            get_links()
            update_output1(self.index)
            update_output2(self.index)
            update_output3(self.index)
        
        def select_anchor(change):
            if self.index == self.nsize - 1:
                next_btn.disabled = True
                analyze_btn.disabled = False
            elif self.index == 0:
                back_btn.disabled = True
                next_btn.disabled = False
            else:
                back_btn.disabled = False
                next_btn.disabled = False
            if change.new == 'None':
                d = tp.compute_drift(self.links[self.index])
                d.loc[0] = [0, 0]
                self.drifts[self.index] = d
                self.dlinks[self.index] = tp.subtract_drift(self.links[self.index].copy(), d)
            elif change.new != 'None' and change.new != None:
                d = tp.compute_drift(
                    self.links[self.index][self.links[self.index]['particle']==int(change.new)]
                    )
                d.loc[0] = [0, 0]
                self.drifts[self.index] = d
                self.dlinks[self.index] = tp.subtract_drift(self.links[self.index].copy(), d)
                        
        def change_labelstracks(change):
            with output1:
                output1.clear_output(True)
                display(HTML('<img src="loading.gif">'))
            update_output1(self.index)
            
        def change_modelcb(change):
            self.usemodel = str(change.new)
            
        def change_shifti(change):
            self.shift_i = str(change.new)
            changeval(change)
        
        @folders_wg.observe
        def change_folder(c):
            submit_btn.disabled = False
        
        @submit_btn.on_click
        def submitbtn_click(b):
            self.nsize = initialwg_list[2].value
            self.usemodel = str(usemodel_wg.value)
            submit_btn.disabled = True
            get_currentranges()
            get_initialsettings()
            loadparams()
            get_imagedata()
            get_locations()
            get_links()
            linkwg_subgroup[4].options= [
                'None',
                *self.links[self.index]['particle'].unique().tolist()
                ]
            linkwg_subgroup[4].value = None
            update_output1(self.index)
            update_output2(self.index)
            update_output3(self.index)
            subsetup_box.layout.visibility = 'visible'
            infotxt = "<p style='font-size:24px'><b>{} at {:.2f} (frames {}-{}) </b></p>".format(
                folders_wg.value,
                self.voltstemps[self.index],
                self.setranges[self.index][0],
                self.setranges[self.index][1]
                )
            infotext_wg.value = infotxt
            if self.anchors[self.index] == None:
                next_btn.disabled = True
            linkwg_subgroup[4].observe(select_anchor, names='value')
            for h in range(len(floatsliderwg_subgroup)):
                floatsliderwg_subgroup[h].observe(changeslider, names='value')
            for i in range(len(paramwg_subgroup)):
                paramwg_subgroup[i].observe(changeparams, names='value')
            for j in range(len(linkwg_subgroup)-1):
                linkwg_subgroup[j].observe(changelinks, names='value')
            for k in range(len(checkboxwg_subgroup)):
                checkboxwg_subgroup[k].observe(change_labelstracks, names='value')
            self.children = [ui_box(1)]
        
        @back_btn.on_click
        def backbtn_click(b):
            if self.index == 1:
                back_btn.disabled = True
            elif self.index == self.nsize - 1:
                next_btn.disabled = False
            saveparams()
            loadingscreen('all')
            self.index += -1
            loadparams()
            infotxt = "<p style='font-size:24px'><b>{} at {:.2f} (frames {}-{}) </b></p>".format(
                folders_wg.value,
                self.voltstemps[self.index],
                self.setranges[self.index][0],
                self.setranges[self.index][1]
                )
            infotext_wg.value = infotxt
            analyze_btn.disabled = True
            get_locations()
            get_links()            
            update_output1(self.index)
            update_output2(self.index)
            update_output3(self.index)
                
        @next_btn.on_click
        def nextbtn_click(b):
            if self.index == self.nsize - 2:
                next_btn.disabled = True
            elif self.index == 0:
                make_dir()
            saveparams()
            save_output()
            self.index += 1
            infotxt = "<p style='font-size:24px'><b>{} at {:.2f} (frames {}-{}) </b></p>".format(
                folders_wg.value,
                self.voltstemps[self.index],
                self.setranges[self.index][0],
                self.setranges[self.index][1]
                )
            infotext_wg.value = infotxt
            loadparams()
            get_locations()
            get_links()            
            update_output1(self.index)
            update_output2(self.index)
            update_output3(self.index)
            if self.anchors[self.index] == None:
                linkwg_subgroup[4].options= [
                    'None',
                    *self.links[self.index]['particle'].unique().tolist()
                    ]
                linkwg_subgroup[4].value = None
                next_btn.disabled = True
            analyze_btn.disabled = True

        @analyze_btn.on_click
        def analyzebtn_click(b):
            MPP = self.imsets[0].nm_per_pixel
            save_output()
            #Method 1 of calculating D: variance of all displacements of Delta_t=1
            for i in range(self.nsize):
                dlc = self.dlinks[i].copy()
                self.displacements1.append(calculate_displacements(MPP, dlc))
            for d in self.displacements1:
                self.D_constants1.append(
                    (d.dx.var() + d.dy.var()) / 4  # r^2 = x^2 + y^2 = 2Dt + 2Dt
                    ) 
                self.mu_hats.append(
                    np.mean(d[['dx', 'dy']], axis=0)
                    )
            #Method 2 of calculating D: linear fit to MSD
            for j in range(self.nsize):
                em0 = tp.emsd(
                    self.dlinks[j],
                    MPP,
                    self.DIFFUSION_TIME,
                    max_lagtime=len(self.imsets[j].frames),
                    detail=True
                    )
                self.em.append(em0)
                self.ed.append([em0['<x>'], em0['<y>']])
                result = linregress(em0.index[:-8] * self.DIFFUSION_TIME, em0['msd'][:-8])
                self.msd_slope.append(result.slope)
                self.msd_intercept.append(result.intercept)
                self.D_constants2.append(result.slope/4)
            with output1:
                output1.clear_output(True)                
                print('...D calculated')
                plot_drift_vectors(MPP)
                print('...drift vectors plotted')
                plot_drift_scalar()
                print('...drift scalar plotted')
                plot_diffusion(heater=checkboxwg_subgroup[2].value)
                print('...diffusion plotted')
                plot_msd()
                print('...msd plotted')
                plot_ed()
                print('...ed plotted')
                plot_v_over_D()
                print('...v_over_D plotted')
                save_toyaml()
                print('...params saved as params.yaml')
                foldername = folders_wg.value+'_{}-{}'.format(self.setranges[0][0], self.setranges[-1][1])
                print('...all files saved to: ')
                print('{}'.format(foldername))
                print('-End-')
            for i in [output2, output3]:
                with i:
                    i.clear_output(True)
                    display(HTML('<img src="loading_complete.gif">'))

        def iou_coef(y_true, y_pred, smooth=1):
            intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
            union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
            iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
            return iou
        
        #initiate trained model
        weighted_loss = dt.losses.flatten(dt.losses.weighted_crossentropy((10,1)))
#         self.model = keras.models.load_model(
#             os.getcwd()+'/motionanalyze_model/iou-0.8610_N1_32-64-128-256_512_lossWL-BCE_bs32_nob60_lrsLIN_doREG.hdf5',
#             custom_objects={
#                 'nd_unet_crossentropy': weighted_loss,
#                 'iou_coef': iou_coef
#                 })
        
        self.nsize = initialwg_list[2].value
        get_currentranges()
        infotxt = "<center><p style='font-size:18px'><font color='#003262'><b>{}</b> {}</p>"
        initialinfo_text1.value = infotxt.format(
            'Current Voltages/Temperatures: ',
            str([round(float(i), 3) for i in self.voltstemps])
            )
        initialinfo_text2.value = infotxt.format('Current Set Ranges: ', str(self.setranges))
        
        if self.index == 0:
            for g in range(len(initialwg_list)):
                initialwg_list[g].observe(changeval, names='value')
            usemodel_wg.observe(change_modelcb, names='value')
            shifti_wg.observe(change_shifti, names='value')
            
        self.children = [ui_box(0)]