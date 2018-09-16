from __future__ import absolute_import

import h5py
import numpy as np
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import measurements
from scipy.spatial import cKDTree
import pylab as plt

import streaks

###################################################
# Currently not using this, but its for detecting 
# local maxima. It's effective when the background 
# isn't too noisy otherwise requires ample smoothing
######################################################
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


def save_peak_param_file( filename, PK_PAR):
    """
    saves a peak parameter file
    """
    with h5py.File(filename,'w') as h5:
        for name,data in PK_PAR.items():
            
            if data is None:
                h5.create_dataset(name, data="_NULL")
            else:
                h5.create_dataset(name, data=data)



def load_peak_param_file( filename):
    """
    Load a peak parameters hdf5 file
    into a python dictionary
    """
    PK_PAR = {}
    with h5py.File(filename, "r") as h5:
        for name in h5.keys():
            data = h5[name].value
            if data == "_NULL":
                data = None
            PK_PAR[name] = data
    
    return PK_PAR

class PeakabooImage:
    """ this class is peak finding on an image"""

    def __init__(self, pk_par, img_sh, how_to_section=None, img=None):
        self.pk_par = pk_par # peakaboo parameters dictionary
        self.img_sh = img_sh # image shape
        
        self.set_mask( pk_par["mask"] )
        
        if img is not None:
            self.set_image( img)
        
        self._set_up_sectioning(how=how_to_section)
    
    def set_mask(self, mask):
        
        if mask is None:
            self.mask = np.ones( self.img_sh, np.bool)
        else:
            self.mask = np.array( mask, np.bool)
            assert( self.mask.shape== self.img_sh) 
  
        self.mask_1d = self.mask.ravel()

    def _set_up_sectioning(self, how=None):
        assert( how in [None, "radial", "fromfile"])
        if how is None:
            self._no_sectioning()
        elif how=="radial":
            self._radial_sectioning()
        elif how=="fromfile":
            self._fromfile_sectioning()
   
        self.how_to_section = how

    def reset_sectioning(self):
        self._set_up_sectioning( how=self.how_to_section)

    def _no_sectioning(self):
        """
        Disables image sectionining such that when peaks are computed
        a global thresholding is applied.
        This is dangerous if the image is not strictly uniform
        as it will lead to biasing (weak peaks will be harder to detect).
        Diffraction images are usually biased in the radial direction
        hence it is best to at least do a radial sectionining.
        Tools to do radial sectionining are embedded in the GUI.
        """
        print("Will not use sectioning")
        self.section_array = np.ones( self.img_sh, np.int) # makes entire image a single section
        self._set_section_idx()

    def _radial_sectioning(self):
        """
        section the image according to radial bins.
        In this way we can detect the peaks as local maxima in
        each radial bin. this is to avoid biasing.. 
        """
#       make sure rbins is defined
        assert( self.pk_par["rbins"] is not None)
        assert( self.pk_par["R"] is not None)

        print("Will use radial sectioning")

#       convert Radial pixel value array to 1D 
        R_1d = self.pk_par["R"].ravel()

#       assign a bin to each radial pixel value according to rbins
        bin_assign = np.digitize( R_1d, self.pk_par["rbins"])

        self.section_array = bin_assign.reshape( self.img_sh) 

        self._set_section_idx()

    def _set_section_idx(self):
        """
        stores 1D index of each pixel in a section
        excluding masked pixels.
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

        assert( self.pk_par["sectioning_array"] is not None)
        
        print("Will use fromfile sectioning")

        self.section_array = self.pk_par["sectioning_array"]
      
        assert( self.section_array.shape==self.img_sh) 
       
        self._set_section_idx()


    def section_thresholding(self):
        
        poss_peaks = np.zeros( self.img_1d.shape, np.bool )
        for section_id in self.section_idx.keys():
            
            inds = self.section_idx[section_id]
            
            self.pts = self.img_1d[ inds]
            
            self._set_possible_peaks_with_median_stats()
            
            is_above_thresh = self.pts > self.pk_par["thresh"]

            is_peak = np.logical_and( self._is_possible_peak, is_above_thresh)
            
            poss_peaks[inds] =  is_peak
            
        self.pk_mask =  poss_peaks.reshape(self.img_sh)


    def _set_possible_peaks_with_median_stats(self):
        """
        Finds outliers in the section using median statistics

        If median deviation is 0, then falls back to  mean statistics
        """
        med = np.median( self.pts )
        diffs = np.sqrt( (self.pts-med)**2 )
        med_diff = np.median( diffs)
        if med_diff == 0:
#           fall back to mean statistics...
            print("fall bac to mean ")
            self._set_possible_peaks_with_mean_stats()
        else: 
            Zscores = 0.6745 * diffs / med_diff 
            self._is_possible_peak = Zscores > self.pk_par["nsigs"]
    
    def _set_possible_peaks_with_mean_stats(self):
        """
        Finds outliers using mean statistics

        Median is priority in the standard Peakaboo usage.
        """
        m = self.pts.mean()
        s = self.pts.std()
        self._is_possible_peak = self.pts > m + self.pk_par["nsigs"] *s
        
        print (m, s, np.sum(self._is_possible_peak ) / float(self.pts.shape[0]))

        

    def section_thresholding_mean(self):
        
        print("Thresholding!")
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
        if not self.pos:
            return
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
        """threshold each array section using median statistics""" 
        
        poss_peaks = np.zeros( self.img_1d.shape, np.bool)
        
        for section_id in self.section_idx.keys():
            
            inds = self.section_idx[section_id]
            
            pts = self.img_1d[inds]
        
            if not np.any( pts > 0):
                continue

            med = np.median( pts )
            diffs = np.sqrt( (pts-med)**2 )
            med_diff = np.median( diffs)
            if med_diff == 0:
                med_diff = np.median( diffs[ pts > 0])
                 
            Zscores = 0.6745 * diffs / med_diff #np.divide( diffs , med_diff , 
                                        #out=np.zeros_like( diffs), 
                                        #where=med_diff != 0)
            print ( "med",med, "med_diff",med_diff,"ptsmean", pts.mean(), "npts", len( pts) ) 
            print ( Zscores)
            print( self.pk_par["nsigs"] ) 
            print ( np.sum( Zscores >= self.pk_par["nsigs"]))
            is_peak = np.logical_and( Zscores >= self.pk_par["nsigs"], pts > self.pk_par["thresh"] )
            
            poss_peaks[inds] =  is_peak
            
        self.pk_mask =  poss_peaks.reshape(self.img_sh)

    def set_image( self, img):
        """
        set the peakaboo image!
        """
        self.img = img

        if self.pk_par["sig_G"] is not None:
            self.filtered_img = gaussian_filter( self.img * self.mask, self.pk_par["sig_G"] )*self.mask
        
        self.set_global_noise() 
        self.img_1d = self.img.copy().ravel()

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


    def set_global_noise(self):
        self.global_noise = self.img[ self.mask ] .std()

    def min_snr_filter(self, bg_zscore=2):
        """
        This method looks locally ar each detected peak
        and analyzes the signal to noise

        bg_zscore , float
            The median absolute deviation threshold for background pixel assignment
            
            All pixels in the vicinity of a detected peak with a zscore below `bg_zscore`
            are considered background.

        First step, a residual sub image is created around each detected peak
            where residual image is the raw sub image (with any filters)
            minus the local background estimation. 
            Local background estimation is actually a 2D plane fit to background pixels
            
            The background pixels are selected using a absolute deviation 
            filter on the sub image: all pixels with absolute deviation below 
            `bg_zscore` are assigned to the background!

            Masked pixels to not enter the computation!

        Second step, The peak mask that was created in self.detect is analyzed locally, 
            and the residual image is used to estimate the peak-in-question SNR
            If snr is below a self.PK_PAR['min_snr'] then the peak is rejected

        """
        if not self.pk_par["filt"]:
            return
        new_pos = []
        new_intens =  []
        
        ypos = np.array([ p[0] for p in self.pos ])
        xpos = np.array([ p[1] for p in self.pos ])
        
        SubImg = streaks.SubImages(self.filtered_img, 
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
            try:
                residual = subimg_proc.get_subtracted_img( bg_zscore)
            except:
                continue
            
            lab,nlab = measurements.label(s_pk.img)
            lab_id = lab[ s.rel_peak]
            
            
            noise = residual.std()
            
            if noise == 0: # NOTE: when does this happen ?
                noise = self.global_noise 
            snr = residual[ lab==lab_id ].max() / noise
           
            if snr < self.pk_par["min_snr"]:
                continue

            new_pos.append( self.pos[i_s] )
            new_intens.append( self.intens[ i_s] )
        
        self.pos = new_pos
        self.intens = new_intens

    def detect(self):
        """
        a method for detecting peaks, trying to speed things up!
        
        Run this after running the thresholding (which sets the pk_mask)!
        """
        self.pos = []
        self.intens = []
        sz = self.pk_par["sz"]
        sig_G = self.pk_par["sig_G"]
        min_conn = self.pk_par["min_conn"]
        max_conn = self.pk_par["max_conn"]
        peak_COM = self.pk_par["peak_COM"]
        
        self.lab_img,_ = measurements.label( self.img*self.pk_mask) 
        
        obs = measurements.find_objects(self.lab_img)
        for i,(sy,sx) in enumerate(obs):
            lab_idx = i+1
            y1 = max(0, sy.start-sz)
            y2 = min(self.img_sh[0], sy.stop+sz)
            x1 = max(0, sx.start-sz)
            x2 = min(self.img_sh[1], sx.stop+sz)    
            
            
            l = self.lab_img[y1:y2,x1:x2]
            
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
            
        self.intens = np.array(self.intens)

    def pk_pos(self, img=None, pk_par=None, reset_mask=False):
        """
        wrapper method for peak detection!
        
        img, np.array image
            if passed, this resets the image
            
        pk_par, peakaboo parameter dictinary
            if passed, this resets the peak parameters

        reset_mask, bool
            If pk_par contains a new mask then reset the mask
            with this parameter, otherwise it wont be set!
            
            This is left up to the user in order to speed up 
            the base peak detection process and not waste
            time comparing new masks with old masks each time
            peak detection is called!

        This function returns a tuple 2 lists:
            pk_positions and pk_intensities

        pk_positions is a list of [ (y1,x1), (y2,x2), ... ]
            where y,x are slow-scan fast-scan peak coors

        pk_intensities is a list of [ I1,  I2, ... ]
            where I is the peak intensity (the maximum intensity in the peak)

        """
        if pk_par is not None:
            self.set_pk_par(pk_par)
        
        if reset_mask:
            self.set_mask( self.pk_par["mask"])
        
        if img is not None:
            self.set_image(img)
        
        self.section_thresholding()
        self.detect()
        
        self.min_dist_filt()
        self.radius_filters()
        self.min_snr_filter()
        
        return self.pos, self.intens

    def set_pk_par(self, pk_par):
        """
        pk_par is a peakaboo parameter dictionary
        """
        self.pk_par = pk_par
        
        self.set_mask( pk_par["mask"])

    def plot_pks( self, ax, pk_par = None, show=True, pause=None):
        """
        ax, matplotlib axis

        pk_par, peakaboo peak parameter dictionary

        show, boolean, if not working in pylab interactive mode, 
            set True to display image

        pause, if working in pylab .draw()  (e.g. psuedo interactive)
            then set to some float value (seconds)
            the plot will pause for this long after drawing...
        """
        if pk_par is not None:
            self.set_pk_par( pk_par)

        m = img[ img > 0].mean()
        s = img[img > 0].std()
        
        ax.imshow( img, vmax=m+5*s, vmin=m-s,  **kwargs) 
        
        pos, intens = self.pk_pos()

        for cent in pos:
            circ = plt.Circle(xy=(cent[1], cent[0]), radius=3, ec='r', fc='none',lw=1)
            ax.add_patch(circ)
        if show:
            plt.show()
        elif pause is not None:
            plt.draw()
            plt.pause(pause)
        
