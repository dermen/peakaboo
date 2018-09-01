from __future__ import absolute_import

import h5py
import numpy as np
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import measurements
from scipy.spatial import cKDTree
import pylab as plt

import streaks

def plot_pks( img, pk=None, show=True,ret_sub=False, **kwargs):
    if pk is None:
        pk,I = pk_pos(img,**kwargs) 
    m = img[ img > 0].mean()
    s = img[img > 0].std()
    plt.imshow( img, vmax=m+5*s, vmin=m-s, cmap='viridis', aspect='equal', interpolation='nearest')
    ax = plt.gca()
    for cent in pk:
        circ = plt.Circle(xy=(cent[1], cent[0]), radius=3, ec='r', fc='none',lw=1)
        ax.add_patch(circ)
    if show:
        plt.show()

    if ret_sub:
        return pk,I
    
def plot_pks_serial( img, pk, delay, ax):
    m = img[ img > 0].mean()
    s = img[img > 0].std()
    while ax.patches:
        _=ax.patches.pop()
    ax.images[0].set_data(img)
    ax.images[0].set_clim(m-s, m+5*s)
    #ax.imshow( img, vmax=m+5*s, vmin=m-s, cmap='viridis', aspect='equal', interpolation='nearest')
    for cent in pk:
        circ = plt.Circle(xy=(cent[1], cent[0]), 
            radius=5, ec='Deeppink', fc='none',lw=1)
        ax.add_patch(circ)
    plt.draw()
    plt.pause(delay)


def detect_peaks(image, neighborhood=None):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    from stack overflow paw detection
    """

    # define an 8-connected neighborhood
    if neighborhood is None:
        neighborhood = generate_binary_structure(2,2)
    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


class PeakabooImage:
    """ this class is peak finding on an image"""

    def __init__(self, pk_par, img_sh, sectioning="radial", img=None):
        self.pk_par = pk_par # peakaboo parameters dictionary
        self.img_sh = img_sh # image shape
        self.L = np.zeros( img_sh, np.int) # for storing labeled regions in image 
        if img is not None:
            self.set_image( img)
        
        self.set_mask( pk_par["mask"] )
        self._set_up_sectioning(how=sectioning)
    
    def set_mask(self, mask):
        
        if mask is None:
            self.mask = np.ones( self.img_sh, np.bool)
        else:
            self.mask = np.array( mask, np.bool)
            assert( self.mask.shape== self.img_sh) 
  
        self.mask_1d = self.mask.ravel()

    def _set_up_sectioning(self, how="radial"):
        assert( how in ["radial", "fromfile"])
        if how=="radial":
            self._radial_sectioning()
  
        elif how=="fromfile":
            self._fromfile_sectioning()
    
    def _radial_sectioning(self):
        """
        section the image according to radial bins
        in this way we can detect the local maxima in
        each radial bin.. this is to avoid biasing.. 
        """
#       make sure rbins is defined
        assert( self.pk_par["rbins"] is not None)
        assert( self.pk_par["R"] is not None)

#       convert Radial pixel value array to 1D 
        R_1d = self.pk_par["R"].ravel()

#       assign a bin to each radial pixel value according to rbins
        bin_assign = np.digitize( R_1d, self.pk_par["rbins"])

        self.section_array = bin_assign.reshape( self.img_sh) 

        self._set_section_idx()

    def _set_section_idx(self):
        """
        stores 1D index of each pixel in a section
        excluding masked pixels..
        """
        
        section_array_1d = self.section_array.ravel()

        section_ids = np.unique( self.section_array)

        self.section_idx = {}
        for section_id in section_ids:
            is_in_section = section_array_1d == section_id
            
            indices = np.logical_and(  is_in_section, self.mask_1d) # mask is True for non-masked
            
            self.section_idx[section_id] = np.where( indices )[0]
       
    def _fromfile_sectioning(self):
        """
        section image from arbitrary file
        that defines sections with integer labels
        """
        assert( self.pk_par["sectioning_file"] is not None)
        
        self.section_array = np.load( self.pk_par["sectioning_file"])
      
        assert( self.section_array.shape==self.img_sh) 
       
        self._set_section_idx()


    def section_thresholding_mean(self):
        
        poss_peaks = np.zeros( self.img_1d.shape, np.bool )

        for section_id in self.section_idx.keys():
            inds = self.section_idx[section_id]

            pts = self.img_1d[inds]

            m = pts.mean()
            s = pts.std()
            

            is_peak = np.logical_and( pts >= m + self.pk_par["nsigs"]*s, 
                                        pts > self.pk_par["thresh"] )

            poss_peaks[ inds] = is_peak

        self.pk_mask =  poss_peaks.reshape( self.img_sh)


    def min_dist_filt( self ):
        """
        chooses between the brightest of two peaks 
        that are too close to one another
        """
        #NOTE: there beith bugs below?
        YXI = np.hstack( ( self.pos, self.intens[:,None]))
        K = cKDTree( YXI[:,:2])    
        pairs = np.array(list( K.query_pairs( self.pk_par["min_dist"])))
        while pairs.size:
            smaller = YXI[:,2][pairs].argmin(1)
            inds = np.unique( [ pairs[i][l] for i,l in enumerate(smaller)] )
            YXI = np.delete(YXI, inds, axis=0)
            K = cKDTree( YXI[:,:2]  )
            pairs = np.array(list( K.query_pairs( self.pk_par["min_dist"])))
        self.pos = YXI[:,:2]
        self.intens = YXI[:,2]


    def section_thresholding_zscore(self):
        
        poss_peaks = np.zeros( self.img_1d.shape, np.bool)
        
        #self.some_peaks = []
        for section_id in self.section_idx.keys():
            
            inds = self.section_idx[section_id]
            
            pts = self.img_1d[inds]
        
            med = np.median( pts )
            diffs = np.sqrt( (pts-med)**2 )
            med_diff = np.median( diffs)
            Zscores = 0.6745 * diffs / med_diff
            
            is_peak = np.logical_and( Zscores >= self.pk_par["nsigs"], pts > self.pk_par["thresh"] )
            poss_peaks[inds] =  is_peak
            
            #self.some_peaks.append( inds[is_peak] )  

        self.pk_mask =  poss_peaks.reshape(self.img_sh)

    def set_image( self, img):
        self.img = img
        self.img_1d = img.copy().ravel()


    def radius_filters(self):
        r_in = self.pk_par["r_in"]
        r_out = self.pk_par["r_out"]
        if r_in is not None or r_out is not None:
            
            assert( self.pk_par["cent"] is not None)
            x_cent, y_cent = self.pk_par["cent"]

            y = np.array([ p[0] for p in self.pos ]).astype(float)
            x = np.array([ p[1] for p in self.pos ]).astype(float)
            r = np.sqrt( (y-y_cent)**2 + (x-x_cent)**2)
            
            if r_in is not None and r_out is not None:
                
                if r_in > r_out:
                    inds = np.logical_and( r > r_out,  r < r_in)
                    inds = np.where( inds)[0]
                else:
                    inds = np.logical_or( r > r_out, r < r_in)
                    inds = np.where( inds)[0]
            
            elif r_in is not None and r_out is None:
                inds = np.where( r < r_in )[0]
            
            elif r_out is not None and r_in is None:
                inds = np.where( r > r_out)[0]
            
            if inds.size:
                self.pos = [self.pos[i] for i in inds]
                self.intens = [self.intens[i] for i in inds]  
            else:
                self.pos,self.intens = [],[]


    def snr_filter2(self, bg_zscore=2):
        if not self.pk_par["filt"]:
            return
        new_pos = []
        new_intens =  []
        
        #ypos, xpos = map( np.array, zip(*self.pos) )
        ypos = np.array([ p[0] for p in self.pos ]).astype(float)
        xpos = np.array([ p[1] for p in self.pos ]).astype(float)
        
        SubImg = streaks.SubImages(self.img, 
            ypos,  xpos, 
            sz=self.pk_par["sz"], 
            mask=self.mask, 
            cent=self.pk_par["cent"])
        
        SubImg_peakmask = streaks.SubImages(self.pk_mask, 
            ypos,  xpos, 
            sz=self.pk_par["sz"], 
            mask=self.mask, 
            cent=self.pk_par["cent"])
        
        for i_s, s in enumerate(SubImg.sub_imgs):
           
            s_pk = SubImg_peakmask.sub_imgs[i_s]

            subimg_proc = streaks.SubImageProcess( s )
            residual = subimg_proc.get_subtracted_img( bg_zscore)
            
            noise = residual.std()
            
            lab,nlab = measurements.label(s_pk.img)
            lab_id = lab[ s.rel_peak]
            
            snr = residual[ lab==lab_id ].max() / noise

            if snr < self.pk_par["min_snr"]:
                continue

            new_pos.append( self.pos[i_s] )
            new_intens.append( self.intens[ i_s] )
        
        self.pos = new_pos
        self.intens = new_intens


    def snr_filter(self):
        if not self.pk_par["filt"]:
            return
        new_pos = []
        new_intens =  []
        
        #ypos, xpos = map( np.array, zip(*self.pos) )
        ypos = np.array([ p[0] for p in self.pos ]).astype(float)
        xpos = np.array([ p[1] for p in self.pos ]).astype(float)
        
        SubImg = streaks.SubImages(self.img, 
            ypos,  xpos, 
            sz=self.pk_par["sz"], 
            mask=self.mask, 
            cent=self.pk_par["cent"])
        
        for i_s, s in enumerate(SubImg.sub_imgs):
            if not s.img.size: # NOTE: do I still need this??
                continue
            
            nconn = streaks.get_nconn_snr_img(s,self.pk_par["min_snr"])
            
            if nconn < 1: 
                continue

            new_pos.append( self.pos[i_s] )
            new_intens.append( self.intens[ i_s] )
        
        self.pos = new_pos
        self.intens = new_intens

    def detect(self):
        self.detect_img = self.img* self.pk_mask
        
        if self.pk_par["sig_G"] is not None:
            self.detect_img = gaussian_filter( self.detect_img, self.pk_par["sig_G"])
    
        self.lab_img, nlab = measurements.label(detect_peaks(self.detect_img))
     
        if self.pk_par["peak_COM"]:
            self.pos = measurements.center_of_mass( self.img, self.lab_img , np.arange( nlab)+1 )
            self.intens = measurements.maximum( self.img, self.lab_img, np.arange( nlab)+1) 
        else:
            self.pos = measurements.maximum_position( self.img, self.lab_img , np.arange( nlab)+1 )
            self.intens = measurements.maximum( self.img, self.lab_img, np.arange( nlab)+1) 

    def detect2(self):
        self.detect_img = self.img* self.pk_mask
        
        if self.pk_par["sig_G"] is not None:
            self.detect_img = gaussian_filter( self.detect_img, self.pk_par["sig_G"])
    
        self.lab_img, nlab = measurements.label(self.detect_img)
     
        if self.pk_par["peak_COM"]:
            self.pos = measurements.center_of_mass( self.img, self.lab_img , np.arange( nlab)+1 )
            self.intens = measurements.maximum( self.img, self.lab_img, np.arange( nlab)+1) 
        else:
            self.pos = measurements.maximum_position( self.img, self.lab_img , np.arange( nlab)+1 )
   
    def detect3(self):
        """
        a method for detecting peaks, trying to speed things up!
        run this after running the thresholding (which sets the pk_mask)!
        """
        self.pos = []
        self.intens = []
        sz = self.pk_par["sz"]
        sig_G = self.pk_par["sig_G"]
        min_conn = self.pk_par["min_conn"]
        max_conn = self.pk_par["max_conn"]
        peak_COM = self.pk_par["peak_COM"]
        _ = measurements.label( self.img*self.pk_mask, output=self.L)
        obs = measurements.find_objects(self.L)
        for i,(sy,sx) in enumerate(obs):
            lab_idx = i+1
            y1 = max(0, sy.start-sz)
            y2 = min(self.img_sh[0], sy.stop+sz)
            x1 = max(0, sx.start-sz)
            x2 = min(self.img_sh[1], sx.stop+sz)    
            
            l = self.L[y1:y2,x1:x2]
            nconn = np.sum(l==(lab_idx))
            if nconn < min_conn:
                continue
            if nconn > max_conn:
                continue
            pix = self.img[ y1:y2, x1:x2]  
            self.intens.append( measurements.maximum( pix, l, lab_idx)) 
            if peak_COM:
                y,x = measurements.center_of_mass( pix, l, lab_idx) 
            else:
                y,x = measurements.maximum_position( pix, l, lab_idx) 
            self.pos.append( (y+y1,x+x1))
        
        self.intens = np.array( self.intens)

    def pk_pos(self, img=None, pk_par=None):
        if pk_par is not None:
            self.set_pk_par(pk_par)
        if img is not None:
            self.set_image(img)
        self.section_thresholding_zscore()
        self.detect3()
        self.min_dist_filt()
        self.radius_filters()
        self.snr_filter2()
        return self.pos, self.intens



def rad_med(I, R, rbins, thresh, mask_val=0):
    
    I1 = I.copy().ravel()
    R1 = R.ravel()

    R1[ I1==mask_val] = 0

    bin_assign = np.digitize( R1, rbins)
    
    for b in np.unique(bin_assign):
        if b ==0:
            continue
        if b == len(rbins):
            continue
        inds = bin_assign==b
        pts = I1[inds]
        
        med = np.median( pts )
        diffs = np.sqrt( (pts-med)**2 )
        med_diff = np.median( diffs)
        Zscores = 0.6745 * diffs / med_diff
        pts[ Zscores < thresh] = 0
        I1[inds] = pts
    return I1.reshape( I.shape)
        
def quick_filter( img, nsigs, mask_val=0): 
    m = img[ img != mask_val].mean()
    s = img[ img != mask_val].std()
    img[ img < m + nsigs*s] = 0
    return img


def pk_pos( img_, make_sparse=True, nsigs=7, sig_G=None, thresh=1, sz=4, min_snr=2.,
    min_conn=-1, max_conn=np.inf, filt=False, min_dist=0, r_in=None, r_out=None,
    mask=None, cent=None, R=None, rbins=None, run_rad_med=False, peak_COM=False, 
    sectioning_fromfile=False, sectioning_file=None):

    sz = int(sz)
    img = img_.copy()
    
    if run_rad_med and rbins is not None:
        img = rad_med( img, R, rbins, nsigs)
    else:
        m = img[ img > 0].mean()
        s = img[ img > 0].std()
        img[ img < m + nsigs*s] = 0
    
    if sig_G is not None:
        img = gaussian_filter( img, sig_G)
    
    lab_img, nlab = measurements.label(detect_peaks(img))
    
    if peak_COM:
        pos = measurements.center_of_mass( img, lab_img , np.arange( nlab)+1 )
        intens = measurements.maximum( img, lab_img, np.arange( nlab)+1) 
    else:
        pos = measurements.maximum_position( img, lab_img , np.arange( nlab)+1 )
        intens = measurements.maximum( img, lab_img, np.arange( nlab)+1) 
        
    good = [ i for i,p in enumerate(pos) if intens[i] > thresh]
    pos = [ pos[i] for i in good]
    intens = [ intens[i] for   i in good]

    if not pos:
        return [],[]
    if r_in is not None or r_out is not None:
        assert( cent is not None)
        y = np.array([ p[0] for p in pos ]).astype(float)
        x = np.array([ p[1] for p in pos ]).astype(float)
        r = np.sqrt( (y-cent[1])**2 + (x-cent[0])**2)
        
        if r_in is not None and r_out is not None:
            
            if r_in > r_out:
                inds = np.logical_and( r > r_out,  r < r_in)
                inds = np.where( inds)[0]
            else:
                inds = np.logical_or( r > r_out, r < r_in)
                inds = np.where( inds)[0]
        
        elif r_in is not None and r_out is None:
            inds = np.where( r < r_in )[0]
        
        elif r_out is not None and r_in is None:
            inds = np.where( r > r_out)[0]
        
        if inds.size:
            pos = [pos[i] for i in inds]
            intens = [intens[i] for i in inds]  
        else:
            return [],[]
    
    npeaks =len(pos)
    if not pos:
        return [],[]
    if min_dist and npeaks>1:
        YXI = np.hstack( (pos, np.array(intens)[:,None]))
        K = cKDTree( YXI[:,:2])        
        pairs = np.array(list( K.query_pairs( min_dist)))
        while pairs.size:
            smaller = YXI[:,2][pairs].argmin(1)
            inds = np.unique( [ pairs[i][l] for i,l in enumerate(smaller)] )
            YXI = np.delete(YXI, inds, axis=0)
            K = cKDTree( YXI[:,:2]  )
            pairs = np.array(list( K.query_pairs( min_dist)))

        pos = YXI[:,:2]
        intens = YXI[:,2]
    if filt:
        new_pos = []
        new_intens =  []
        ypos,xpos = map( np.array, zip(*pos) )
        SubImg = streaks.SubImages(img_.copy(), ypos,  
            xpos, sz=sz,mask=mask, cent=cent)
        
        for i_s, s in enumerate(SubImg.sub_imgs):
            if not s.img.size: # NOTE: do I still need this??
                continue
            
            
            nconn = streaks.get_nconn_snr_img(s,min_snr)
            
            if nconn < 1:
                continue

            if nconn < min_conn:
                continue
            if nconn > max_conn:
                continue

            new_pos.append( pos[i_s] )
            new_intens.append( intens[ i_s] )
        
        pos = new_pos
        intens = new_intens
    
    return pos, intens



def load_peak_param_file( filename):
    PK_PAR = {}
    with h5py.File(filename, "r") as h5:
        for name in h5.keys():
            data = h5[name].value
            print (name, data )
            if data == "_NULL":
                data = None
            PK_PAR[name] = data
    
    return PK_PAR




