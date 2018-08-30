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

import numpy as np
import h5py

from ImageViewer import ImageViewer

frpk = {'padx': 5, 'pady': 5}
btnstyle = {'highlightbackground':'black'}

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
                psana_run_number=None, psana_event_number=None ,
                nav_frame=None, image_frame=None):

        self.nav_frame = nav_frame
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
            assert( psana_event_number is not None)

        self.counter = 0
        self.indices = np.arange( self.imgs.N)
        
        self._set_idx_fname_path()
        self._set_image(init=True)
        self.xl = (None, None)
        self.yl = (None, None)
       
        self._file_nav_buttons()

    def _set_idx_fname_path(self):
        self.idx = self.indices[self.counter]

    
    def _set_image(self, init=False):
        dset = self.imgs 
        self.img = dset[self.idx]  
        if init:
            if self.image_frame is None:
                im_fr= tk.Toplevel(self.master)
            else:
                im_fr = self.image_frame
            self.IV = ImageViewer(im_fr, self.img, attached=False) 
            self.IV.pack( fill=tk.BOTH, expand=tk.YES)
            self.fig = self.IV.fig
            self.ax = self.IV.ax

    def _set_xy_limits(self):
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        self.ax.set_xlim(xl)
        self.ax.set_ylim(yl)

    def _remove_patches(self, patches):
        for p in patches:
            self.ax.patches.remove(p)
        
    def _add_patches( self, patches):
        for p in patches:
            self.ax.add_patch( p)

    def _file_nav_buttons(self):

        if self.nav_frame is None:
            nav_frame = tk.Toplevel( self.master)
        else:
            nav_frame = self.nav_frame
        
        button_frame0 = tk.Frame(nav_frame, bg='black', highlightbackground="#00fa32", highlightthickness=1)
        button_frame0.pack(side=tk.TOP,  **frpk)

        tk.Label( button_frame0, text="File navigation", fg="#00fa32",bg='black' ,  font= 'Helvetica 24 bold')\
            .pack(side=tk.TOP, expand=tk.YES,)
    
        self.img_info_text= "File: %s; image: %d/%d"
        self.file_info_lab = tk.Label( button_frame0, 
            text=self.img_info_text%( self.imgs.filename_i, self.imgs.shot_i, self.imgs.N_i-1  ),
            fg="#00fa32", bg='black')
        self.file_info_lab.pack(side=tk.TOP)
        button_frame = tk.Frame(nav_frame, bg='black')
        button_frame.pack(side=tk.TOP)

        prev_button100 = tk.Button(button_frame0,
                                   text='-100',
                                   command=lambda: self._prev(100), **btnstyle)
        prev_button100.pack(side=tk.LEFT, expand=tk.NO, **frpk)
        prev_button10 = tk.Button(button_frame0,
                                  text='-10',
                                  command=lambda: self._prev(10), **btnstyle)
        prev_button10.pack(side=tk.LEFT, expand=tk.NO, **frpk)
        prev_button1 = tk.Button(button_frame0,
                                 text='-1',
                                 command=lambda: self._prev(1), **btnstyle)
        prev_button1.pack(side=tk.LEFT, expand=tk.NO, **frpk)

        next_button1 = tk.Button(button_frame0,
                                 text='+1',
                                 command=lambda: self._next(1), **btnstyle)
        next_button1.pack(side=tk.LEFT, expand=tk.NO, **frpk)
        next_button10 = tk.Button(button_frame0,
                                  text='+10',
                                  command=lambda: self._next(10), **btnstyle)
        next_button10.pack(side=tk.LEFT, expand=tk.NO, **frpk)
        next_button100 = tk.Button(button_frame0,
                                   text='+100',
                                   command=lambda: self._next(100), **btnstyle)
        next_button100.pack(side=tk.LEFT, expand=tk.NO, **frpk)

    def _next(self, increment):
        self.counter += increment
        if self.counter >= len(self.indices):
            self.counter = self.counter - increment
            self.counter = len( self.indices)-1 
        self.inc_func()    

    def _prev(self, increment):
        self.counter = self.counter - increment
        if self.counter < 0:
            self.counter = 0
        self.inc_func()
    
    def _update_img_info_text(self):
        text=self.img_info_text%( self.imgs.filename_i, self.imgs.shot_i, self.imgs.N_i-1  )
        self.file_info_lab.config(text=text)








