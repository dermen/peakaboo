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


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    from stack overflow paw detection
    """

    # define an 8-connected neighborhood
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

def rad_med(I,R, rbins, thresh, mask_val=0):
    I1 = I.copy().ravel()
    R1 = R.ravel()

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
        

def pk_pos( img_, make_sparse=True, nsigs=7, sig_G=None, thresh=1, sz=4, min_snr=2.,
    min_conn=-1, max_conn=np.inf, filt=False, min_dist=0, r_in=None, r_out=None,
    mask=None, cent=None, R=None, rbins=None, run_rad_med=False, peak_COM=False):

    sz = int(sz)
    img = img_.copy()
    #img[ img <  thresh] = 0
    if run_rad_med and rbins is not None:
        img = rad_med( img, R, rbins, nsigs)
    else:
        m = img[ img > 0].mean()
        s = img[ img > 0].std()
        img[ img < m + nsigs*s] = 0
    
    if sig_G is not None:
        img = gaussian_filter( img, sig_G)
    lab_img, nlab = measurements.label(detect_peaks(img))
    locs = measurements.find_objects(lab_img)
    
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
            #SubProc = streak_peak.SubImageProcess(s)

            #nconn = SubProc.get_nconnect_cent(zscore_sig=min_snr)
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

