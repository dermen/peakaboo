from __future__ import absolute_import

import numpy as np
import pylab as plt
import h5py
try:
    from itertools import izip
except ImportError:
    izip = zip

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import center_of_mass
from scipy import optimize
import glob
from scipy.ndimage.filters import gaussian_filter as gauss_filt

try:
    from astride import Streak
    has_astride=True
except ImportError:
    print("Does not have astride installed.")
    has_astride=False

class SubImageProcess:
    
    def __init__( self, sub_img):
        self.sub_img = sub_img
        self.img = self.sub_img.img
        self.img_1d = self.img.ravel()
        self.mask = self.sub_img.pixmask
    
    def _get_BG_pixels(self, zscore):
        
        outs = is_outlier( self.img_1d, zscore)
        outs = outs.reshape ( self.img.shape)
        
        outs = np.logical_or( outs, ~self.mask)

        self.BG_mask = ~outs
        BG = (~outs) * self.img
        self.BG_pixels = BG
        
        self.bg_value = self.img[~outs].mean()
        self.sigma_value = self.img[~outs].std()

    def set_tilt_plane(self, zscore) :
        self._get_BG_pixels(zscore)
        
        Y,X = np.indices(self.img.shape)
        YY,XX = Y.ravel(), X.ravel()

        x,y,z = X[self.BG_mask], Y[self.BG_mask], \
            self.img[self.BG_mask]
        guess = np.array([np.ones_like(x), x, y ] ) .T
        coeff, r, rank, s = np.linalg.lstsq(guess, z)
        ev = (coeff[0] + coeff[1]*XX + coeff[2]*YY )
        self.tilt_plane = ev.reshape( self.img.shape)

    def get_subtracted_img(self, zscore):
        self.set_tilt_plane(zscore)
        return self.img - self.tilt_plane
        
    def get_residual_outlier_img(self, zscore_bg, zscore_sig):
        residual_img = self.get_subtracted_img( zscore_bg)
        residual_img_1d = sub_img.ravel()
        outs = is_outlier(residual_img_1d, zscore_sig)
        outs = outs.reshape( self.img.shape)
        return outs

    def get_nconnect_cent(self, zscore_bg=2, zscore_sig=3):
        resid = self.get_residual_outlier_img(zscore_bg, \
            zscore_sig)
        lab,nlab = label(resid)
        if lab[self.sub_img.rel_peak] == 0:
            return 0
        nconn = np.sum( lab == lab[self.sub_img.rel_peak]  )
        return nconn

def get_nconn_snr_img( sub_img, min_snr, zscore_bg=2, 
        structure=None):
    
    if structure is None:
        structure = generate_binary_structure( 2,2)

    snr_img = get_snr_img( sub_img, min_snr, zscore_bg)
    
    lab,nlab = label(snr_img, structure=structure)
    if lab[sub_img.rel_peak] == 0:
        return  0
    nconn = np.sum( lab == lab[sub_img.rel_peak]  )
    return nconn

def get_snr_img( sub_img, min_snr, zscore_bg=2):
    sp = SubImageProcess( sub_img)
    resid = sp.get_subtracted_img( zscore_bg )
    noise = np.sqrt( resid[ sp.BG_mask].std()**2 \
        + resid[ ~sp.BG_mask].std() **2 )
    snr_img =  (resid / noise) * (~sp.BG_mask)
    return snr_img >= min_snr

def get_nconnect_cent( sub_img, zscore_bg, zscore_sig):
    sp = SubImageProcess( sub_img)
    return sp.get_nconnect_cent( zscore_bg, zscore_sig)


def tilting_plane(img, zscore=2 ):
    outs = is_outlier( img.ravel(), zscore)
    outs = outs.reshape ( img.shape)
    BG = (~outs) * img
    Y,X = np.indices( img.shape)
    YY,XX = Y.ravel(), X.ravel()

    x,y,z = X[~outs], Y[~outs], img[~outs]
    guess = array([np.ones_like(x), x, y ] ) .T
    coeff, r, rank, s = np.linalg.lstsq(guess, z)
    ev = (coeff[0] + coeff[1]*XX + coeff[2]*YY )
    return ev.reshape( img.shape)

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fit_gauss( peak_pro, xdata, width ):
    mu     = xdata[ peak_pro.argmax() ] 
    sig    = width
    amp    =  peak_pro.ptp() /2.
    offset   =  peak_pro.min()

    guess =( amp, mu,  sig )

    try:
        fit,success = optimize.curve_fit( gauss, 
                        xdata=xdata, ydata=peak_pro, p0 =guess )
    except RuntimeError: 
        return None
    return fit, success,gauss(xdata, *fit)

class Imgs_from_dataframe:
    def __init__(self, df ):
        fnames = df.cxi_fname.unique()
        self.h5s = [ h5py.File(f,'r') for f in fnames]
        
        gb = df.groupby('cxi_fname')

        self.N = sum( [h[data_path].shape[0] for h in self.h5s])
        
        self.data_path = data_path
        self._make_index()
    def _make_index(self):
        self.I = {}
        count = 0
        for i,h in enumerate(self.h5s):
            N_data = h[self.data_path].shape[0]
            for j in range( N_data):
                self.I[count] = {'file_i':i, 'shot_i':j}
                count += 1
    def __getitem__(self,i):
        file_i = self.I[i]['file_i']
        shot_i = self.I[i]['shot_i']
        return self.h5s[file_i][self.data_path][shot_i]


########
# process
class SubImages:
    def __init__(self, img, y,x, sz, 
        mask=None, cent=None):

        if cent is None:
            self.cent =  (.5* img.shape[0], .5* img.shape[1] ) 
        else:
            self.cent = cent
        self.img = img
        if mask is None:
            self.mask = np.ones( self.img.shape, bool)
        else:
            assert( mask.shape == self.img.shape)
            self.mask = mask.astype(bool)
        self.y = y
        self.x = x
        self.sz = sz
        self._make_sub_imgs()

    def _make_sub_imgs(self):
        ########
        yo,xo=self.cent
        sz = self.sz
        img = self.img
        y = self.y
        x = self.x
        ########
        ymax = img.shape[0]
        xmax = img.shape[1]

        #y,x = map(np.array, zip(*pk))

        self.sub_imgs = []
        bounds = {'slow':[0,0,0], 'fast':[0,0,0]}
        for i,j in izip( x,y):
            j2 = int( min( ymax,j+sz  ) )
            j1 = int( max( 0,j-sz  ) )
            
            i2 = int( min( xmax,i+sz  ) )
            i1 = int( max( 0,i-sz  ) )
            bounds['slow'] = [j1,j,j2]
            bounds['fast'] = [i1,i,i2]

            r = np.sqrt( (j-yo)**2 + (i-xo)**2)
            sub = SubImage( img[ j1:j2, i1:i2 ] , 
                bounds, 
                r, 
                mask=self.mask[ j1:j2, i1:i2])
            self.sub_imgs.append(sub) 
    
    def integrate(self, **kwargs):
        for s in self.sub_imgs:
            s.integrate_streak(**kwargs)
    
    def integrate_preds(self, **kwargs):
        for s in self.sub_imgs:
            s.integrate_pred_streak(**kwargs)

class SubImage:
    def __init__(self, img, bounds=None, radius=None, 
        blind_radius=6, mask=None):

        self.img = img
        if mask is None:
            self.pixmask = np.ones( self.img.shape, bool)
        else:
            self.pixmask=mask
        if bounds is not None and radius is not None:
            self.peak = int(bounds['slow'][1]), \
                int(bounds['fast'][1])
            self.rel_peak = int(bounds['slow'][1] -\
                                bounds['slow'][0]), \
                            int(bounds['fast'][1] - \
                                bounds['fast'][0] )
            self.radius = radius
            self.subimg_corner = np.array( [ bounds['slow'][0], bounds['fast'][0] ] )
        else:
            self.peak = None
            self.radius = None 
            self.corner = None
        Y,X = np.indices( img.shape)
        self.pix_pts = np.array( zip(X.ravel(), 
            Y.ravel() ) )

        self.R = np.sqrt( (Y-self.rel_peak[0])**2 \
            + (X-self.rel_peak[1])**2 )
    
        self.blind_region = self.R < blind_radius

    def _set_streak_mask( self, sig_G=None, **kwargs):
        
        assert( has_astride)

        if sig_G is not None:
            streak = Streak(gauss_filt(self.img, sig_G),
                output_path='.',
                **kwargs)
            streak.detect()
        else:
            streak = Streak(self.img,
                output_path='.',
                **kwargs)
            streak.detect()
        self.streak = streak
        edges = streak.streaks
        if not edges:
            self.has_streak=False
            self.mask =np.ones(self.img.shape, bool  ) 
            self.circulaity= []
            self.streak_centers = []
            self.roundness = []
            self.areas = []
            return
        verts = [ np.vstack(( edge['x'], edge['y'])).T 
                        for edge in edges]
        paths = [ plt.mpl.path.Path(v) 
                for v in verts ]
        
        self.streak_centers = [ [edge['y_center'], \
            edge['x_center']] for edge in edges ]
        
        self.circularity = [   edge['shape_factor'] 
            for edge in edges]
        self.roundness = [ edge['radius_deviation'] 
            for edge in edges]
        self.areas = [ edge['area'] for edge in edges]
        
        contains = np.vstack( [ p.contains_points(self.pix_pts) 
            for p in paths ])
        self.streak_masks = [ c.reshape( self.img.shape) 
            for c in contains]
        mask = np.any( contains,0).reshape( self.img.shape)
        self.mask = np.logical_not(mask)
        self.has_streak=True

    def get_streak_mask(self, sig_G=None, **kwargs):
        self._set_streak_mask(sig_G=sig_G, **kwargs)
        return self.mask

    def get_stats( self, sig_G=None, **kwargs):
        img = self.img
        
        self._set_streak_mask(sig_G =sig_G, **kwargs)

        self.bg_mask = (self.mask==1)* self.pixmask
        self.bg_pix = img[self.bg_mask].astype(float)
        bg = self.bg_pix.mean()
        #self.bg_pix -= bg
        noise = (img[ self.bg_mask ]-bg).std()
        
        self.bg_sub_sig = noise
        self.bg_sub_mn = (img[ self.bg_mask ]-bg).mean()

        return bg, noise

    def integrate_streak( self, min_conn=0, 
        max_conn=np.inf, **kwargs):
        
        #assert( self.peak is not None and \
        #    self.radius is not None)
        
        bg, noise = self.get_stats(**kwargs)
        regions, n = label( ~self.mask)
        residual = self.img - bg
        cent_region = regions[self.rel_peak[0], 
            self.rel_peak[1]]
        connect = np.sum( regions==cent_region)
        self.sig_mask = regions==cent_region
        #if min_conn < connect < max_conn:
        counts = residual[ regions==cent_region].mean()
        #counts = residual[ regions==cent_region].sum()
        #else:
        #    counts = np.nan
        self.counts = counts
        self.bg = bg
        self.sigma = noise
        self.peak_region = regions==cent_region
        self.N_connected = connect
        self.lab_dist=0
        #return counts, bg , noise, residual, connect
    
    def integrate_pred_streak( self, 
            sig_G=None, 
            dist_cut=np.inf, 
            try_gauss=True,
            **kwargs):
        
        #assert( self.peak is not None and \
        #    self.radius is not None)
        bg, noise = self.get_stats(sig_G=sig_G, **kwargs)
        
        #regions, n = label( ~self.mask)
        
        if not self.areas: #n == 0:
            self.integrate_blind(bg,noise)
            return

        lab_pos = np.array( self.streak_centers)
        lab_dists = np.sqrt( np.sum( \
            (lab_pos - np.array( self.rel_peak) )**2,1) )
        idx = np.argmin( lab_dists)
        
        cent_region_ma = self.streak_masks[ idx]
        self.area = self.areas[idx]
        self.round = self.roundness[idx]
        self.circ = self.circularity[idx]
        self.streak_COM = self.streak_centers[idx] + np.array( self.subimg_corner)
        #self.dist2 = lab_dists2[idx]

        #cent_region = u_labs[ np.argmin( lab_dists)] # +1
        peak_dist = lab_dists.min()
        self.lab_dist = peak_dist #lab_dists.min()
        if peak_dist > dist_cut:
            if try_gauss:
                self.integrate_blind(bg,noise)
            else:
                self.integrate_blind2(bg,noise)
            return 
        residual = self.img - bg
        
        connect = np.sum( cent_region_ma)
        #if min_conn < connect < max_conn:
        #else:
        #    counts = np.nan
        
        self.sig_mask = (cent_region_ma)*self.pixmask
        self.sig_pix = residual[self.sig_mask]
        
        #counts = residual[ regions==cent_region].sum()
        #self.counts = self.sig_pix.sum()
        self.counts = self.sig_pix.sum() #mean()
        self.bg = bg
        self.sigma = noise
        self.N_connected = connect
        self.has_signal = True
        self.used_gauss=False
    
    def integrate_blind(self, bg, noise, nbins=20, thresh=2., use_gauss=False):
        
        residual = self.img-bg
        
        self.sig_mask = self.blind_region * \
            self.mask*self.pixmask

        self.sig_pix =  residual[self.sig_mask]
       
#       fit gaussian
        a,b = np.histogram( self.bg_pix-bg, 
            bins=nbins)
        bb = b[:-1]*.5 + b[1:]*.5
        fit = fit_gauss( a, bb, 10.)
        
        if fit:
            self.used_gauss=True
            m = fit[0][1]
            s = fit[0][2]
        else:
            self.used_gauss=False
            m = self.bg_sub_mn
            s = self.bg_sub_sig
        
        sig_pixels = self.sig_pix[ self.sig_pix  > m +  thresh * s]
        
        if sig_pixels.size:
            self.counts = sig_pixels.sum() #mean()
            self.has_signal=True
        else:
            self.counts = m
            self.has_signal=False
        
        #if self.counts < 0:
        #    self.counts = 0
        
        self.bg = bg
        self.sigma = s
        self.N_connected = np.nan
        self.area = np.nan
        self.round = np.nan
        self.circ = np.nan
        self.streak_COM = self.peak #np.nan

    def integrate_blind2(self, bg, noise):
        
        residual = self.img-bg
        
        self.sig_mask = self.blind_region * \
            self.mask*self.pixmask

        self.sig_pix =  residual[self.sig_mask]
        self.counts = self.sig_pix.mean()
        #if self.counts < 0:
        #    self.counts = 0
        self.bg = bg
        self.sigma = noise
        self.N_connected = np.nan
        self.lab_dist = np.nan
        self.area = np.nan
        self.round = np.nan
        self.circ= np.nan
        self.streak_COM = self.peak
        self.has_signal=True
        self.used_gauss=False


def make_sub_imgs( img, pk, sz):
    ymax = img.shape[0]
    xmax = img.shape[1]

    y,x = map(np.array, zip(*pk))

    sub_imgs = []
    for i,j in izip( x,y):
        j2 = int( min( ymax,j+sz  ) )
        j1 = int( max( 0,j-sz  ) )
        
        i2 = int( min( xmax,i+sz  ) )
        i1 = int( max( 0,i-sz  ) )
        sub_imgs.append( img[ j1:j2, i1:i2 ] )
 
    return sub_imgs


def find_edges(img, sig_G=0, **kwargs):
    assert( has_astride)
    streak = Streak(gaussian_filter(img,sig_G), **kwargs)
    streak.detect()
    return streak.streaks

def plot_streak( img, ax, sig_G_lst=[0], **kwargs):
    assert( has_astride)
    m = np.median( img[ img >0])
    s = np.std(img[ img >0])
    ax.imshow( img, vmax=m+4*s, vmin=m-s, cmap='viridis')
    for sig_G in sig_G_lst:
        streak = Streak(gaussian_filter(img,sig_G),
            output_path='.',
            **kwargs)
        streak.detect()
        edges = streak.streaks
        if not edges:
            return 0
        verts = [ np.vstack(( edge['x'], edge['y'])).T 
                        for edge in edges]
        paths = [ plt.mpl.path.Path(v) 
                for v in verts ]
        for p in  paths:
            patch = plt.mpl.patches.PathPatch(p, 
                facecolor='none', 
                lw=2, 
                edgecolor='Deeppink')
            ax.add_patch(patch)

def get_streak_mask( img, pix_pts, **kwargs):
    #streak = Streak(gaussian_filter(img,sig_G), 
    assert( has_astride)
    streak = Streak(img,output_path='.',
        **kwargs)
    streak.detect()
    edges = streak.streaks
    if not edges:
        return np.ones(img.shape, bool  )
    
    verts = [ np.vstack(( edge['x'], edge['y'])).T 
                    for edge in edges]
    paths = [ plt.mpl.path.Path(v) 
            for v in verts ]
    contains = np.vstack( [ p.contains_points(pix_pts) 
        for p in paths ])
    mask = np.any( contains,0).reshape( img_sh)

    return np.logical_not(mask)

def get_stats( img, pix_pts, **kwargs):
    mask = get_streak_mask(img, pix_pts, **kwargs)
    #assert( mask is not None)
    bg = img[ mask == 1].mean()
    sig1 = img[mask ==1 ].std()
    sig2 = img[ mask==0].std()
    noise = sqrt( sig1**2 + sig2**2 )
    return mask, bg, noise

def make_snr_img( img, pix_pts, **kwargs):
    mask, bg, noise = get_stats( img, pix_pts, **kwargs)
    snr_img = (img-bg)/noise
    return snr_img


def integrate_streak( img, pix_pts, min_conn=0, max_conn=np.inf, **kwargs):
    mask, bg, noise = get_stats( img, pix_pts, **kwargs)
    regions, n = label( ~mask)
    sub_img = img - bg
    #if n ==1:
    #counts = sub_img[ regions==1 ].sum()
    #else:
    sh = sub_img.shape
    cent_region = regions[ sh[0]/2, sh[1]/2]
    connect = sum( regions==cent_region)
    if min_conn < connect < max_conn:
        counts = sub_img[ regions==cent_region].sum()
    else:
        counts = np.nan
    return counts, bg , noise, sub_img, connect

def integrate_streak_list( imgs, pix_pts, min_conn=0, max_conn=np.inf, **kwargs):
    output = np.zeros( ( len( imgs), 4  ) )
    for i_img, img in enumerate(imgs):
        counts, bg, noise, S,conn = integrate_streak( img, pix_pts, min_conn, max_conn, **pars)
        output[i_img,0] =  counts
        output[i_img,1] =  bg
        output[i_img,2] =  noise
        output[i_img,3] =  conn
    return output


def next_subs(im,y,x, sub_sz=10):
    pk = map( np.array, zip(y, x) )
    subs = make_sub_imgs( im, pk, sub_sz )
    return subs

def write_cxi_peaks( h5, peaks_path, pkX, pkY, pkI):
    
    npeaks = np.array( [len(x) for x in pkX] )
    max_n = max(npeaks)
    Nimg = len( pkX )
    
    data_x = np.zeros((Nimg, max_n), dtype=np.float32)
    data_y = np.zeros_like(data_x)
    data_I = np.zeros_like(data_x)

    for i in xrange( Nimg): 
        n = npeaks[i]
        data_x[i,:n] = pkX[i]
        data_y[i,:n] = pkY[i]
        data_I[i,:n] = pkI[i]
    
    peaks = h5.create_group(peaks_path)
    peaks.create_dataset( 'nPeaks' , data=npeaks)
    peaks.create_dataset( 'peakXPosRaw', data=data_x )
    peaks.create_dataset( 'peakYPosRaw', data=data_y )
    peaks.create_dataset( 'peakTotalIntensity', data=data_I ) 



def is_outlier(points, thresh=3.5):
    """
    http://stackoverflow.com/a/22357811/2077270
    
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points-median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745*diff / med_abs_deviation

    return modified_z_score > thresh


