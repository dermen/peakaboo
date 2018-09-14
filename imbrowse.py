"""
classes for navigating through a series of images
"""

try: 
    import Tkinter as tk
    import tkFileDialog
except ImportError:
    import tkinter as tk
    from tkinter import filedialog as tkFileDialog

import glob
from matplotlib.collections import PatchCollection

import numpy as np
import h5py

from zoomIm5 import ImageViewer

try:
    import psana
    has_psana = True
except ImportError:
    has_psana = False
    print("Did not find psana; call peakaboo with a psana python distribution.")

frpk = {'padx': 5, 'pady': 5}
btnstyle = {'highlightbackground':'black'}


class PsanaImages:
    def __init__(self, exp, run, detector_name, N_events=10000, codes=None):
        """
        exp: experiment string
        run: run number
        detector_name: detecrtor string
        N_events: interact w this many events , up to total events in run
            if -1, processes all events found
        """
        assert( has_psana)
       
        self.run = run
        self.exp = exp
        self.N_events = self.N = N_events
        self.codes = codes
        self.ds = psana.DataSource("exp=%s:run=%s"%(exp,run))
        self.detnames = [ d for sl in psana.DetNames() for d in sl]
        self.env = self.ds.env()
        
        assert(  detector_name  in self.detnames)
       
        self.code_dets = [ psana.Detector(d, self.env) 
                    for d in self.detnames if d.startswith("evr") ]
        
        self.events = self.ds.events() #this is a generator!

        self.detector_name = detector_name
        self.Detector = psana.Detector( self.detector_name, self.env)

        self.event = self.events.next()
        self.event_index = 0

    def __getitem__(self, i):
        """
        only allow forward indexing of psana events... for now
        """
        assert( self.event_index <= i)
        
        while self.event_index < i:
            self.event = self.events.next()
            if self.event is None:
                continue
            self.event_index += 1
            print("skipping image %d / %d "%( self.event_index, i))
        img = self._get_image()
        
        while img is None:
            print("None image")
            self.event = self.events.next()
            if self.event is None:
                continue
            #self.event_index += 1
            img = self._get_image()
        
        return img

    def _get_image( self):
        self.codes = []
        for cdet in self.code_dets:
            c = cdet.eventCodes(self.event)
            if c is not None:
                self.codes += c
        self.event_codes = list(set( self.codes))
        self.evr_str = " ".join([str(c) for c in self.event_codes ])
        self.filename_i = "run: %d; exp: %s; evr: %s"\
                %(self.run, self.exp, self.evr_str )
        self.shot_i = self.event_index
        self.N_i = self.N_events

        img = self.Detector.image( self.event)
        if img is None:
            self.filename_i = "Nonetype image"
        else:
            self.filename_i = "run: %d; exp: %s; evr: %s"\
                %(self.run, self.exp, self.evr_str )
        
        return img
        

class multi_h5s_img:
    """this class takes a list ofhdf5 filenames with image data
    (e.g. CXIDB) and loads the images as a single iterator"""
    
    def __init__(self, fnames, data_path):
        """
        fnames, the list of fnames
        data_path, the path to the data in each file
        """
        self.h5s = [ h5py.File(f,'r') for f in fnames]
        self.N = sum( [h[data_path].shape[0] for h in self.h5s])
        self.data_path = data_path
        self._make_index()
    
    def _make_index(self):
        """ makes a hash index that returns filename, 
            dataset index, and global index"""
        self.I = {}
        count = 0
        for i,h in enumerate(self.h5s):
            N_data = h[self.data_path].shape[0]
            for j in range( N_data):
                self.I[count] = {'file_i':i, 'shot_i':j, 'N':N_data}
                count += 1
    
    def __getitem__(self,i):
        """ returns an image and sets some attributes of that image""" 
        file_i = self.I[i]['file_i'] # h5 file handle
        
        self.shot_i = self.I[i]['shot_i'] # dataset index
        self.N_i = self.I[i]['N'] # global index
        self.filename_i = self.h5s[file_i].filename # h5 filename
        
        return self.h5s[file_i][self.data_path][self.shot_i]



class BrowseImages:
    """a high level class for browsing images"""
    
    def __init__( self , master,  increment_function, how="files", 
                h5_fnames=None, h5_images_path=None, 
                psana_run_number=None, psana_experiment_name=None , 
                psana_detector_name=None, 
                image_nav_frame=None, image_frame=None, image_strides = None):

        if image_strides is None:
            self.image_strides =[-100, -10,-1,1,10, 100]
        else:
            self.image_strides = sorted( images_strides)

        self.image_nav_frame = image_nav_frame
        self.image_frame = image_frame
        self.master = master
        self.inc_func = increment_function

        assert (how in ['files' , 'psana'] )
        
        if how=='files':
            assert( h5_fnames is not None)
            assert( h5_images_path is not None)
            fname_paths = glob.glob( h5_fnames)
            self.imgs = multi_h5s_img( fname_paths, h5_images_path)
        
        elif how =='psana':
            assert( psana_run_number is not None)
            assert( psana_experiment_name is not None)
            assert( psana_detector_name is not None)
            self.imgs = PsanaImages( psana_experiment_name, 
                psana_run_number, psana_detector_name) 

        self.counter = 0
        self.indices = np.arange( self.imgs.N)
        
        self._set_idx_fname_path()
        self._set_image(init=True)
        self.xl = (None, None)
        self.yl = (None, None)
       
        self._image_nav_buttons()


    def _set_idx_fname_path(self):
        self.idx = self.indices[self.counter]

    def _set_image(self, init=False):
        dset = self.imgs 
        if init:
            if self.image_frame is None:
                self.im_view_fr= tk.Toplevel(self.master)
                #self.im_view_fr.resizable(False,False)
            else:
                self.im_view_fr = self.image_frame
            self.IV = ImageViewer(self.im_view_fr, dset[self.idx]) 
            self.IV.pack( expand=tk.NO)
            self.fig = self.IV.fig
            self.ax = self.IV.ax
        else:
            self.IV.img = dset[self.idx]

    def _set_xy_limits(self):
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        self.ax.set_xlim(xl)
        self.ax.set_ylim(yl)

    def _remove_patches(self, patches):
        for p in patches:
            #self.IV.ax.patches.remove(p)
            p.remove() #self.IV.ax.remove(p)
    
    def _remove_patch_collections(self, label=None):
        if label is None:
            return
        for p in self.IV.ax.collections:
            if p.get_label()==label:
                p.remove()
                print("Removed %s from axis"%label)
        
    def _add_patches( self, patches, label=None, **kwargs):
        pc = PatchCollection( patches, label=label, **kwargs)
        self.IV.ax.add_collection( pc)
        
        #self.IV._im.figure.canvas.draw()
        #print self.IV.ax.patches 
        #self.IV.fig.canvas.draw()
        #self.IV.update_master_image()
        #self.IV.fig.canvas.blit(self.ax.bbox)
        #self.IV.fig.canvas.flush_events()

    def _image_nav_buttons(self):

        if self.image_nav_frame is None:
            image_nav_frame = tk.Toplevel( self.master)
        else:
            image_nav_frame = self.image_nav_frame
        
        button_frame0 = tk.Frame(image_nav_frame, bg='black', highlightbackground="#00fa32", highlightthickness=2)
        button_frame0.pack(side=tk.TOP,  fill=tk.X, expand=tk.NO, **frpk)

        tk.Label( button_frame0, text="Image navigation", fg="#00fa32",bg='black' ,  font= 'Helvetica 14 bold')\
            .pack(side=tk.TOP, expand=tk.YES,)
    
        self.img_info_text= "File: %s; image: %d/%d"
        self.img_info_lab = tk.Label( button_frame0, 
            text=self.img_info_text%( self.imgs.filename_i, self.imgs.shot_i, self.imgs.N_i-1  ),
            fg="#00fa32", bg='black')
        self.img_info_lab.pack(side=tk.TOP)

#       navigation strides
        self.nav_buttons = {}
        for stride in self.image_strides:
            if stride > 0:
                self.nav_buttons[stride] = tk.Button(button_frame0,
                    text='%+d'%stride,
                    command=lambda increment=np.abs(stride): self._next(increment) , **btnstyle)
            else:
                self.nav_buttons[stride] = tk.Button(button_frame0,
                    text='%+d'%stride,
                    command=lambda increment=np.abs(stride): self._prev(increment) , **btnstyle)
            
            self.nav_buttons[stride].pack(side=tk.LEFT, expand=tk.YES, fill=tk.X, **frpk)

    def _next(self, increment):
        self.counter += increment
        if self.counter >= len(self.indices):
            #self.counter = self.counter - increment
            self.counter = len( self.indices)-1 
        self.inc_func()    

    def _prev(self, increment):
        self.counter = self.counter - increment
        if self.counter < 0:
            self.counter = 0
        self.inc_func()
    
    def _update_img_info_text(self):
        text=self.img_info_text%( self.imgs.filename_i, self.imgs.shot_i, self.imgs.N_i-1  )
        self.img_info_lab.config(text=text)


