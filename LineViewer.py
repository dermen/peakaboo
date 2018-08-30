#!/usr/bin/python
try: 
    import Tkinter as tk
except ImportError:
    import tkinter as tk
import sys
import re
import os
import pandas
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import pylab as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
btnstyle = {'font': 'Helvetica 10 bold', 
            'activebackground': 'green', 'activeforeground': 'white',
            'relief': tk.RAISED, 'highlightcolor':'red'}  # , 'bd':5}
labstyle = {'font': 'Helvetica 14 bold', 'bg': 'snow', 'fg': 'black',
            'activebackground': 'green', 'activeforeground': 'white'}

fr = {'bg': None}
frpk = {'padx': 5, 'pady': 5}

class LineViewer(tk.Frame):
    """
    Main line viewer ; 
    """
    def __init__(self, master, line_data, xlabel="x-axis", 
        ylabel="y-axis", 
         *args, **kwargs):
        
        tk.Frame.__init__(self, *args, **kwargs) #master,  background='white') 
        self.master = master
        
        self.line_frame = tk.Frame( self.master,  **fr )
        self.line_frame.pack( side=tk.TOP)
        
        self.ymin = self.ymax = self.xmin = self.xmax = None
        self.ylabel=ylabel
        self.xlabel=xlabel
        #plot the line 
        self.line_data = line_data

        
        self._create_figure()    
        self._add_line()
        self._setup_canvas()
        
    def _create_figure(self):
        #self.fig, self.ax = plt.subplots(1,1)
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.1, .1, .99, .99])
    
    def _add_line(self):
        self._line = self.ax.plot(
            self.line_data[0], 
            self.line_data[1], 
            lw=2)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        self._update_lims()
    
    def _update_lims(self):
        yvals = self.line_data[1]
        ymax = max(yvals) + 0.2*max(yvals) 
        ymin = min(yvals) - 0.2*min(yvals) 
        self.ax.set_ylim( ymin, ymax) 
        
        xvals = self.line_data[0]
        xmax = max(xvals) + 0.2*max(xvals) 
        xmin = min(xvals) - 0.2*min(xvals) 
        self.ax.set_xlim( xmin, xmax) 
        
    def update_data( self, line_data):
        self.line_data= line_data
        self.ax.lines[0].set_data( *self.line_data)
        self._update_lims()
        #self.ax.figure.canvas.draw()
        #plt.pause(0.1)

    def _setup_canvas(self):
        #toplvl= tk.Toplevel(self.master)
        self.disp_frame = tk.Frame(self.master)
        self.disp_frame.pack(side=tk.TOP, expand=1, fill=tk.BOTH, **frpk)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.disp_frame) 
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, 
            expand=1, **frpk)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, 
            self.disp_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, 
            fill=tk.BOTH, expand=1, **frpk)

        #self.canvas.draw()
        


#if __name__=="__main__":
#    
#    line_data = [range(10), range(10)]
#    root    = tk.Tk()
#    line_fr =  LineViewer(root, line_data=line_data)
#    line_fr.pack()
#    
#    plt.pause(2)
#    line_fr.update_data( [range(100), np.random.random(100) ])    
#    root.mainloop()


    

