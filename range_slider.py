#!/usr/bin/python
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

import numpy as np

labstyle={'bg':'black', 'fg':'#00fa32'}

fr = {'bg':None}
frpk = {'padx':5, 'pady':5}
class RangeSlider(tk.Frame):
    '''
    range slider 
    Ideas: https://stackoverflow.com/a/6789351/2077270
    '''
    def __init__(self, master, color='#00fa32', 
            range_slider_len=400, 
            minval=None, maxval=None, *args, **kwargs):
       
        tk.Frame.__init__(self, master,  *args, **kwargs)
        
        self.color = color
      
        self.master = master

#       set max and min vals...
        self._create_min_max_entry()
        self.minval = minval
        self.maxval = maxval
        self.dmin = minval
        self.dmax = maxval
        self._set_min_max_vals()

#       slide bounds

        self.range_slider_len = range_slider_len
        self.range_slider_height= 30
        self.range_slider_token_height=20
        self.range_slider_token_halfheight=10
        self.token_text_vertical_position=7.5
        self.range_slider_token_vertical_offset = 12.5
        self.range_slider_token_width = 5
        self.range_slider_token_halfwidth = 2.5
        self.range_slider_token_LHS_offset = 15
        self.range_slider_token_RHS_offset = 15
        self.text_offset_LHS = 2.5
        self.text_offset_RHS = -2.5
        self.range_slider_value_range = self.range_slider_len \
            -2*self.range_slider_token_width - self.range_slider_token_LHS_offset\
            -self.range_slider_token_RHS_offset
        
        self.range_slider_maxRHS = self.range_slider_len - self.range_slider_token_halfwidth - self.range_slider_token_RHS_offset

        self.range_slider_minLHS = self.range_slider_token_halfwidth + self.range_slider_token_LHS_offset
        
#       create a canvas for drawing the range-slider...
        self.canvas_frame = tk.Frame( self.master, bg="black", ) 
        self.canvas = tk.Canvas(self.canvas_frame, width=self.range_slider_len, 
            height=self.range_slider_height, bg='black', highlightbackground="black")
       
#       reate horizontal line sepcifying the slider token track
        self.canvas.create_line( ( self.range_slider_token_LHS_offset, 
            self.range_slider_token_vertical_offset+self.range_slider_token_halfheight, 
            self.range_slider_len-self.range_slider_token_RHS_offset, 
            self.range_slider_token_vertical_offset+self.range_slider_token_halfheight), fill=self.color) 
        
#       this data is used to keep track the slider components as they are dragged
        self._drag_data = {"x": 0, "y": 0, "item": None}

#       create a couple of movable sliders
#       SLIDER TOKENS 
        i1 = self._create_token((self.range_slider_minLHS, 
            self.range_slider_token_vertical_offset+self.range_slider_token_halfheight), 
            self.color)
        i2 = self._create_token((self.range_slider_maxRHS,
            self.range_slider_token_vertical_offset+self.range_slider_token_halfheight), 
            self.color)
        self.items = [i1, i2] # store the IDs

        self._set_text_label()

        # add bindings for clicking, dragging and releasing over
        # any object with the "token" tag
        self.canvas.tag_bind("token", "<ButtonPress-1>", self.on_token_press)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self.on_token_release)
        self.canvas.tag_bind("token", "<B1-Motion>", self.on_token_motion)
        
        self._pack_widgets()

    def _pack_widgets(self):
        self.entry_minval.pack(side=tk.LEFT,)# expand=tk.YES)
        self.canvas_frame.pack(side=tk.LEFT,expand=tk.NO, padx=10)
        self.canvas.pack(expand=tk.NO, fill=tk.BOTH,side=tk.LEFT,)
        self.entry_maxval.pack(side=tk.LEFT, expand=tk.NO)

    def _create_min_max_entry(self):
        self.entry_minval = tk.Entry(self.master, bd=2, 
            relief=tk.GROOVE, 
            width=5)
        self.entry_maxval = tk.Entry(self.master, bd=2, relief=tk.GROOVE, width=5)
        self.entry_minval.bind('<Return>', self._get_min_max_vals )
        self.entry_maxval.bind('<Return>', self._get_min_max_vals )

        self.entry_maxval.config( bg='black', fg=self.color, insertbackground=self.color)
        self.entry_minval.config( bg='black', fg=self.color, insertbackground=self.color)


    def _get_min_max_vals(self ,event):
        dmin = float( self.entry_minval.get() )
        dmax = float( self.entry_maxval.get() )
        
        assert( dmin < dmax)
        
        # move the left item
        if self.dmin != dmin:
            self.minval = dmin
            self.dmin = dmin
            
            left_item = self.items[0]
            (left_x,left_y) = (self.range_slider_minLHS, 
                self.range_slider_token_vertical_offset+self.range_slider_token_halfheight)
            x1,y1,x2,y2 = (left_x-self.range_slider_token_halfwidth, 
                        left_y-self.range_slider_token_halfheight, 
                        left_x+self.range_slider_token_halfwidth, 
                        left_y+self.range_slider_token_halfheight)
            self.canvas.coords( left_item, (x1,y1,x2,y2))
            #   move the label
            self.canvas.itemconfig( self.minvaltext, text="%.2f"%self.minval)
            self.canvas.coords( self.minvaltext, 
                        (self.range_slider_minLHS-self.text_offset_LHS, 
                        self.token_text_vertical_position))
           
        # move the right item
        if self.dmax != dmax:
            self.dmax = dmax
            self.maxval = dmax
            right_item = self.items[1]
            (right_x, right_y) = (self.range_slider_maxRHS,
                self.range_slider_token_vertical_offset+self.range_slider_token_halfheight)
            x1,y1,x2,y2 = (right_x-self.range_slider_token_halfwidth, 
                        right_y-self.range_slider_token_halfheight, 
                        right_x+self.range_slider_token_halfwidth, 
                        right_y+self.range_slider_token_halfheight)
            self.canvas.coords( right_item, (x1,y1,x2,y2))
        
       
            # move the label
            self.canvas.itemconfig( self.maxvaltext, text="%.2f"%self.maxval)
            
            self.canvas.coords( self.maxvaltext, (self.range_slider_maxRHS+self.text_offset_RHS, 
                    self.token_text_vertical_position))

    def _set_min_max_vals(self): 
        self.entry_minval.delete(0,tk.END)
        self.entry_minval.insert(0,"%.1f"%self.minval)
        self.entry_maxval.delete(0,tk.END)
        self.entry_maxval.insert(0,"%.1f"%self.maxval)
        
    def _set_text_label(self):
#       TEXT LABEL
        self.minvaltext = self.canvas.create_text( 
            self.range_slider_minLHS-self.text_offset_LHS, 
            self.token_text_vertical_position, text="%.2f"%self.dmin, font="Helvetica 11 bold", fill="white")
        self.maxvaltext = self.canvas.create_text( 
            self.range_slider_maxRHS+self.text_offset_RHS, 
            self.token_text_vertical_position, text="%.2f"%self.dmax,font="Helvetica 11 bold", fill="white")
        
    def _create_token(self, coord, color):
        '''Create a token at the given coordinate in the given color'''
        (x,y) = coord
        return self.canvas.create_rectangle(x-self.range_slider_token_halfwidth, 
            y-self.range_slider_token_halfheight, 
            x+self.range_slider_token_halfwidth, 
            y+self.range_slider_token_halfheight, 
            outline=color, fill=color, tags="token")

    def on_token_press(self, event):
        '''Begining drag of an object'''
        # record the item and its location
        self._drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x

    def on_token_release(self, event):
        '''End drag of an object'''
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0

    def on_token_motion(self, event):
        '''Handle dragging of an object'''
        clicked_item = self._drag_data["item"]
        coors = self.canvas.coords(clicked_item)
        xleft, ybottom, xright, ytop = coors
        clicked_x = .5* (xleft + xright)
        clicked_y = .5* (ybottom + ytop)
        
        other_item = [i for i in self.items if i != clicked_item][0]
        other_coors = self.canvas.coords(other_item)
        other_x = .5*(other_coors[0] + other_coors[2])
        other_y = .5*(other_coors[1]+ other_coors[3])
        if clicked_x > other_x:
            #   RIGHT HAND SIDE ITEM
            delta_x = event.x - self._drag_data["x"]
            
            x_new = min( clicked_x + delta_x, self.range_slider_maxRHS) 
            x_new = max( x_new, other_x+self.range_slider_token_width) 

            self.canvas.coords( clicked_item, (x_new-self.range_slider_token_halfwidth, 
                clicked_y-self.range_slider_token_halfheight, 
                    x_new+self.range_slider_token_halfwidth, 
                    clicked_y+self.range_slider_token_halfheight))
            self.canvas.coords( self.maxvaltext, (x_new+self.text_offset_LHS,
                self.token_text_vertical_position))
            self.canvas.itemconfig( self.maxvaltext, text="%.2f"%self.maxval)
            
            new_drag = min( self.range_slider_maxRHS,  event.x)
            new_drag = max(  other_x , new_drag)
            self._drag_data["x"] = new_drag 
            
            self.maxval = ( (x_new) / self.range_slider_value_range ) \
                * ( self.dmax - self.dmin ) + self.dmin
            
        else:
            #   LEFT HAND SIDE ITEM
            delta_x = event.x - self._drag_data["x"]
            
            x_new = max( clicked_x + delta_x, self.range_slider_minLHS) 
            x_new = min( x_new, other_x-self.range_slider_token_width) 
            
            self.canvas.coords( clicked_item, 
                (x_new-self.range_slider_token_halfwidth, 
                    clicked_y-self.range_slider_token_halfheight, 
                    x_new+self.range_slider_token_halfwidth, 
                    clicked_y+self.range_slider_token_halfheight))
            self.canvas.coords( self.minvaltext, (x_new-self.text_offset_RHS, 
                self.token_text_vertical_position))
            self.canvas.itemconfig( self.minvaltext, text="%.2f"%self.minval)
            new_drag = max( self.range_slider_minLHS,  event.x)
            new_drag = min(  other_x , new_drag)
            self._drag_data["x"] = new_drag 
            
            self.minval = ((x_new) / self.range_slider_value_range ) \
                * ( self.dmax - self.dmin ) + self.dmin

def main():
    root = tk.Tk()
    RS = RangeSlider(root,color='blue', minval=0, maxval=1000, range_slider_len=800,) 
    RS.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH) 
    root.mainloop()
    
if __name__ == "__main__":
    main()
