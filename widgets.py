try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

class LabeledEntry(tk.Frame):
    
    def __init__(self, master, label, variable, label_width=None, entry_width=None, *args, **kwargs):
        
        tk.Frame.__init__(self.master, *args, **kwargs)
        self.variable = variable
        self.text = text
        self.master = master
        self.colorA = '#00fa32'
        self.colorB = 'black'
        self.entrystyle = {'fg':self.colorA, 'bg':self.colorB, 
                        'selectforeground':self.colorA,
                        'insertbackground':self.colorB}
        
        tk.Label(self.master, text=text, fg=self.colorA, bg=self.colorB, width=label_width).pack(side=tk.LEFT)
        
        self.entry = tk.Entry( self.master, textvariable=variable, width=entry_width, **style)
        self.entry.pack(side=tk.LEFT)
    
   
class LabeledCheckbutton(tk.Frame):
    
    def __init__(self, master, label, command,  label_width=None, *args, **kwargs):
        
        tk.Frame.__init__(self, *args, **kwargs)
        self.master = master

        self.colorA = '#00fa32'
        self.colorB = 'black'

        self.variable = tk.IntVar(self.master)
        self.check_button = tk.Checkbutton(self.master, 
                        variable=self.variable, 
                        command=command, 
                        bg=self.colorB)
        self.check_button.pack(side=tk.LEFT)
        tk.Label(self.master, text=label, 
            fg=self.colorA, bg=self.colorB, 
            width=label_width)\
            .pack(side=tk.LEFT)


