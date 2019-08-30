#!/usr/bin/env python
# coding: utf-8

# # Todo
# 
# 
# * Build PCA from much larger set of sky observations
# * work off real spectra (not sky_spectrum)
# 
# 

# # Notes
# * all experiments are currently in pca_sky1.ipynb and pca_sky3.ipynb 
# * this notebook is meant for production

# In[2]:


COMMANDLINE = True


# In[3]:


if not COMMANDLINE:
    # go wide screen
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))

    # Next two lines force automatic reload of loaded module. Convenient if
    # one is still fiddeling with them.
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


# In[7]:


import sys

import numpy as np

from astropy.io import fits
from astropy.stats import biweight_location
from sklearn.decomposition import PCA
import pickle
import os
import glob
from matplotlib import pyplot as plt
if not COMMANDLINE:
    get_ipython().run_line_magic('matplotlib', 'inline')

from collections import OrderedDict
import spectrum
from numpy import polyfit,polyval


# In[8]:


def load_skys(ff, which="sky_spectrum", normalize=False):
    """
    Loads bunch of spectra (2D) from set of input file names.
    """
    skys = OrderedDict()
    wws  = OrderedDict()
    #shotids = OrderedDict()
    N = len(ff)
    for i,f in enumerate(ff):
        if i % 100 == 0:
            print("loading {} out of {}.".format(i,N))
        shotid = f.split("/")[-3]
        exp = f.split("/")[-2]
        try:
            ww,rb = pickle.load( open(f,'rb'), encoding='iso-8859-1' )
            skys[(shotid,exp)] = rb[which]/rb["fiber_to_fiber"] 
            wws[(shotid,exp)] = ww
            
            
            
            #print("+ start wl = ", ww[0], "A", "end wl = ", ww[-1], "A", "len(ww) ", len(ww), " shape rb ", [rb[k].shape for k in rb])
            #shotids[(shotid,exp)] = f
        except:
            print("Error loading {}.".format(f))
            pass
        
        if normalize:
            # NEW try to normalize by mean
            skys[(shotid,exp)][np.isnan(skys[(shotid,exp)])] = 0.
            skys[(shotid,exp)] = (skys[(shotid,exp)].T /np.mean( skys[(shotid,exp)], axis=1 )).T
    print("start wl = ", ww[0], "A", "end wl = ", ww[-1], "A", "len(ww) ", len(ww), " shape rb ", rb[which].shape)
    return wws, skys


# In[9]:


def build_XA(IFU, ww, skys, wstart, wend, amps):
    # Select referece source
    # here we will use as A  the beiweight location (~ mean) from the entire IFU 
    XA = []
    for k in skys[(IFU, amps[0])]:
        amps_data = []
        for amp in amps:
            if k in skys[(IFU,amp)]:
                amps_data.append( skys[(IFU,amp)][k] )
        stack = np.vstack(amps_data)
        bloc = biweight_location( stack, axis=0) 
        XA.append(bloc)

    XA = np.array(XA)

    if False:
        # hack to homogenize lengths, the rebinning does make sure
        # that the wavelength grid always stars at the same wavelength
        # but not necessarey, end at the same ( there may be a few pixel more or less)
        #N = np.min([XA.shape[1], XB.shape[2], ww.shape[0]])
        N = np.min([XA.shape[1], ww.shape[0]])
        ww = ww[:N]
        XA = XA[:,:N]

    # can't have nans
    XA[np.isnan(XA)] = 0.

    ii = (ww >= wstart) * (ww <= wend)
    wwcut = ww[ii]
    XAcut = XA[:,ii]

    return wwcut, XAcut


# In[10]:


def build_XB(IFU, amp, ww, skys, wstart, wend):
    B = (IFU, amp) # here we select, which IFU and amp we build the 
                     # PCA sky for
        
    # first build big array that holds all skys in B
    XB = np.array( [skys[B][k] for k in skys[B] ] )

    # trim in wavelength space
    ii = (ww >= wstart) * (ww <= wend)
    
    # homogenize the length
    XB[np.isnan(XB)] = 0.
    # and select a spectral subrange as set above
    XBcut = XB[:,:,ii]

    return wwcut, XBcut


# In[11]:



def homogenize(wws, skys, start0=None, stop0=None, length0=None, skyx0=None, skyy0=None):
    print(start0, stop0, length0, skyx0, skyy0)
    #Check that all wavelength arrays and spectra have same dimensions and sampling
    def check(wws, skys, start0, stop0, length0, skyx0, skyy0):
        starts,stops,lenghts,skyx,skyy = [],[],[],[],[]
        for ifu,amp in wws:
            for shot,exp in wws[(ifu,amp)]:
                sky = skys[(ifu,amp)][(shot,exp)]
                ww = wws[(ifu,amp)][(shot,exp)]
                starts.append(ww[0])
                stops.append(ww[-1])
                lenghts.append(len(ww))
                skyx.append(sky.shape[1])
                skyy.append(sky.shape[0])

        if start0 == None:
            start0 = starts[0]
        if stop0 == None:
            stop0 = stops[0]
        if length0 == None:
            length0 = lenghts[0]
        if skyx0 == None:
            skyx0 = skyx[0]
        if skyy0 == None:
            skyy0 = skyy[0]
            
        same_starts = np.all( np.array(starts) == start0 )
        same_stops = np.all( np.array(stops) == stop0 )
        same_lenghts = np.all( np.array(lenghts) == length0 )
        same_skyx = np.all( np.array(skyx) == skyx0)
        same_skyy = np.all( np.array(skyy) == skyy0 )
        
        print( "All wavelength arrays have same start wl: {} ({})".format( same_starts,  np.unique(starts)) )
        print( "All wavelength arrays have same stop wl: {} ({})".format( same_stops, np.unique(stops)) )
        print( "All wavelength arrays have same length: {} ({})".format( same_lenghts, np.unique(lenghts)) )
        print( "All sky arrays have same x size: {} ({})".format( same_skyx, np.unique(skyx) ) )
        print( "All sky arrays have same y size: {} ({})" .format( same_skyy, np.unique(skyy) ) )
        return ww, starts, stops, lenghts, skyx, skyy, same_starts, same_stops, same_lenghts, same_skyx, same_skyy
    
    ww, starts, stops, lenghts, skyx, skyy, same_starts, same_stops, same_lenghts, same_skyx, same_skyy =         check(wws, skys, start0, stop0, length0, skyx0, skyy0)
    
    
    # If different in wavelength extent, fix.
    if same_starts and not (same_stops and same_lenghts and same_skyx):
        print("Fixing difference in wavelength extent.")
        
        if start0 != None:
            starts.append(start0)
        max_startwl = np.max( starts )
        if stop0 != None:
            stops.append(stop0)
        min_stopwl = np.min( stops )
        if length0 != None:
            lenghts.append(length0)
            
        N = np.min(lenghts)
        for ifu,amp in wws:
            for shot,exp in wws[(ifu,amp)]:
                sky = skys[(ifu,amp)][(shot,exp)]
                ww = wws[(ifu,amp)][(shot,exp)]
                
                wws[(ifu,amp)][(shot,exp)] = ww[:N]
                skys[(ifu,amp)][(shot,exp)] = sky[:,:N] 
                
        # check again, now they better be the same
        ww, starts, stops, lenghts, skyx, skyy, same_starts, same_stops, same_lenghts, same_skyx, same_skyy =             check(wws, skys, start0, stop0, length0, skyx0, skyy0)

    
    # if all good return wavelength array that is good for all
    if same_starts and same_stops and same_lenghts and same_skyx:
            
        return ww, starts[0], stops[0], lenghts[0], skyx[0],  skyy[0]
    else:
        print("All are not equal. Stopping here.")
        sys.exit(1)


# In[156]:


def save_sky(IFU, amp , k, wwcut, pca_sky, dir_rebin):
    pattern="{}/{}/{}/multi_???_{}_???_{}_rebin.pickle"
    shotid, exp = k

    _pattern = pattern.format(dir_rebin, shotid, exp, IFU, amp)
    ff = glob.glob(_pattern)
    if not len(ff) == 1:
        print("ERROR: Did not find files like {}".format(_pattern))
        return
    fname = ff[0]

    h,t = os.path.split(fname)
    pca_fname = os.path.join(h,"pca_" + t)

    ww,rb = pickle.load( open(fname,'rb'), encoding='iso-8859-1' )

    N = np.min([ww.shape[0], rb['sky_subtracted'].shape[1], rb["fiber_to_fiber"].shape[1], rb["sky_spectrum"].shape[1]])
    rb["fiber_to_fiber"] = rb["fiber_to_fiber"][:,:N]
    rb["sky_subtracted"] = rb["sky_subtracted"][:,:N]
    ww = ww[:N]
    rb["sky_spectrum"] = rb["sky_spectrum"][:,:N]
    rb["pca_sky_spectrum"] = rb["sky_spectrum"].copy()
    ii = (ww >= wwcut[0]) * (ww <= wwcut[-1])
    rb["pca_sky_spectrum"][:,ii] = pca_sky * rb["fiber_to_fiber"][:,ii]


    rb['pca_sky_subtracted'] = rb['sky_subtracted'] + rb['sky_spectrum'] - rb['pca_sky_spectrum']

    pickle.dump(  ( ww,rb), open(pca_fname,'wb') , protocol=2   )
    print("Wrote: ", pca_fname)
        


# In[226]:


IFU = "022"; amps=["LL", "LU"]; amps_skysub=["LL"]
IFU = "022"; amps=["RL", "RU"]; amps_skysub=["RL"]

IFU = "022"; amps=["LL", "LU"]; amps_skysub=["LU"]
IFU = "022"; amps=["RL", "RU"]; amps_skysub=["RU"]

#IFU = "022"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "022"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "022"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "023"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "023"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "023"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "023"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "033"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "033"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "033"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "033"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "034"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "034"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "034"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "034"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "035"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "035"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "035"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "035"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "036"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "036"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "036"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "036"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "037"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "037"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "037"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "037"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "042"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "042"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "042"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "042"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "043"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "043"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "043"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "043"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "044"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "044"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "044"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "044"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "045"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "045"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "045"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "045"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "046"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "046"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "046"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "046"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "073"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "073"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "073"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "073"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "074"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "074"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "074"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "074"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "076"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "076"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "076"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "076"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "083"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "083"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "083"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "083"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "084"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "084"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "084"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "084"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "085"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "085"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "085"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "085"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "086"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "086"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "086"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "086"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "093"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "093"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "093"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "093"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "094"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "094"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "094"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "094"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "095"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "095"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "095"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "096"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "096"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "096"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "096"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "103"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "103"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "103"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "103"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "104"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "104"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "104"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "104"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "105"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "105"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "105"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "105"; amps=["RU"]; amps_skysub=["RU"]
#IFU = "106"; amps=["LL"]; amps_skysub=["LL"]
#IFU = "106"; amps=["LU"]; amps_skysub=["LU"]
#IFU = "106"; amps=["RL"]; amps_skysub=["RL"]
#IFU = "106"; amps=["RU"]; amps_skysub=["RU"]


# In[227]:



fn_shotlist_pca = "data/shotlist_pca.txt"
fn_shotlist_skyrecon = "data/shotlist_skyrecon.txt"

dir_rebin="data/rebin"
# selct wavelength subrange
wstart = 3495.
wend = 5493.
MEANSHIFT = True

# how many PCA components do we want to maintain?
n_components = 20
USEPCA = False

kappa = .8 # for sky outlier clipping


TEST_MOCK_EMISSION = False


# In[228]:


if COMMANDLINE:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ifu', default='022', required=True)
    parser.add_argument('-a', '--Aamps', nargs='+', help='<Required> Source amps.', required=True)
    parser.add_argument('-b', '--Bamp',             help='<Required> Target amp.', required=True)

    parser.add_argument('--shotlist_pca', default="shotlist_pca.txt")
    parser.add_argument('--shotlist_skyrecon', default="shotlist_skyrecon.txt")
    parser.add_argument('--dir_rebin', default="pca_test/rebin")
    parser.add_argument('--sky_kappa', default=0.8, type=float)
    parser.add_argument('--ncomp', default=20, type=int)

    args = parser.parse_args()

    IFU=args.ifu
    fn_shotlist_pca = args.shotlist_pca
    fn_shotlist_skyrecon = args.shotlist_skyrecon
    dir_rebin = args.dir_rebin
    amps = args.Aamps
    amps_skysub = [args.Bamp]

    n_components = args.ncomp
    kappa = args.sky_kappa # for sky outlier clipping


# In[229]:


# Source shotlist for sky PCA component computation
# all nights of cosmos repeats
with open(fn_shotlist_pca, 'r') as f:
    s = f.read()
shotlist_PCA = s.split()

with open(fn_shotlist_skyrecon, 'r') as f:
    s = f.read()
shotlist_skyrecon = s.split()


# # First part: Compute PCA components for A and pseudo-components for B based on a large number of skys

# In[230]:


# load rebinned data

def load_skys_for_shotlist(IFU, shotlist, amps):
    # load all skys for given list of shots
    # this newer version makes sure that there is always date for all four amplifieres
    ff = OrderedDict()
    for amp in amps:
        ff[amp] = []
    exposures = ['exp01','exp02','exp03']
    for shot in shotlist:
        if shot.startswith("#"):
            continue
        # go through all exposures and make sure all amps have data
        # discard exposure if not
        for e in exposures:
            amp_files = OrderedDict()
            for amp in amps:
                pattern = "{}/{}/{}/multi_???_{}_???_{}_rebin.pickle".format(dir_rebin, shot, e, IFU, amp)
                #print(pattern)
                fff = glob.glob(pattern)
                if len(fff) == 0:
                    print("No file found like {}. Check rebin dir.".format(pattern))
                    continue
                amp_files[amp] = fff[0]
        
            if amp in amp_files.keys() and all( [  amp_files[amp] != '' for amp in amps] ):
                for amp in amps:
                    ff[amp] += [amp_files[amp]]
            else:
                print("WARNING: for {} exp {}, not all four amps have data, dropping ....".format(shot, e))
    skys = OrderedDict()
    wws  = OrderedDict()
    
    for amp in amps:
        wws[(IFU,amp)],skys[(IFU,amp)] = load_skys(ff[amp],which="sky_spectrum")

    return wws,skys


wws, skys = load_skys_for_shotlist(IFU, shotlist_PCA, amps)


# In[231]:


_shotids = [k for k in skys[(IFU,amps[0])] ]

# Check that all wavelength arrays and spectra have same dimensions and sampling
ww, start0, stop0, length0, skyx0, skyy0 = homogenize(wws, skys)

# build data matrix A
wwcut, _XAcut = build_XA(IFU, ww, skys, wstart, wend, amps)


# In[232]:


start0, stop0, length0, skyx0, skyy0


# In[233]:


# reject outlier spectra
from astropy.stats import biweight_midvariance, biweight_location

mm = biweight_location( _XAcut[:,450:600], axis=1)
s = np.sqrt( biweight_midvariance(mm) )
m = biweight_location(mm) 

start, stop = m-kappa*s, m+kappa*s

#start, stop = 130., 230.

jj = (mm > start) * (mm < stop)
print("{} of {} survive cut.".format(np.sum(jj),len(jj)))

if True:
    
    f = plt.figure(figsize=[20,4])
    ax = plt.subplot(1,2,1)
    plt.hist(mm,bins=np.arange(0,400,10))

    plt.axvline(start,c='b')
    plt.axvline(stop,c='b')
    plt.xlabel("mean counts")
    plt.ylabel("N")

    ax = plt.subplot(1,2,2)
    plt.imshow((_XAcut.T/mm).T[jj], vmin=.6,vmax=2.4, origin="bottom")
    plt.savefig("{}/histcut_{}_{}.pdf".format(dir_rebin, IFU, "".join(amps)))
XAcut = _XAcut[jj]
shotids = np.array(_shotids)[jj]


# normilze (usually done by scikit.learn's PCA method also, but useful for plotting etc.)
MA = np.mean(XAcut,axis=0)
if MEANSHIFT:
    XAmean = XAcut - MA
else:
    XAmean = XAcut


# In[234]:


# PCA computation for A

plotfn = "{}/pca_explvarA_{}_{}.pdf".format(dir_rebin, IFU,"".join(amps))

    
PLOT_EXPL_VAR = True
if USEPCA:
    if not MEANSHIFT:
        print("ERROR: PCA must use mean shift") 
    else:
        pcaA = PCA(n_components=n_components)
        pcaA.fit(XAmean)
        if PLOT_EXPL_VAR:
            f = plt.figure(figsize=[7,7])
            # explained_variance vs. n components
            plt.plot(pcaA.explained_variance_ratio_, 'o')  
            plt.xlabel("N")
            plt.ylabel("PCA explained variance ratio")
            plt.yscale('log')
            #print(pcaA.singular_values_)
            f.tight_layout()
            plt.savefig(plotfn)
else:
    # SVD computation for A
    from sklearn.decomposition import TruncatedSVD
    pcaA = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    pcaA.fit(XAmean)
    
    if PLOT_EXPL_VAR:
        # explained_variance vs. n components
        f = plt.figure(figsize=[7,4])
        plt.plot(pcaA.explained_variance_ratio_, 'o')  
        plt.xlabel("N")
        plt.ylabel("SVD explained variance ratio")
        plt.yscale('log')
        #print(pcaA.singular_values_) 
        f.tight_layout()
            
        plt.savefig(plotfn)


# In[235]:


# project pca componets onto
#  mean shifted input spectra
ccA = np.inner(XAmean, pcaA.components_)

# reconstruct pca comonents through linear combination
rcA = np.matmul( XAmean.T, ccA).T

# they won't be normalized yet
for i,cA in enumerate(rcA):
    rcA[i] = rcA[i]/np.linalg.norm(rcA[i])


# In[236]:


# make sure the result is the same
if not (np.abs( rcA - pcaA.components_) < 1e-6).all():
    print("WARNING: Recontructed PCA components are not (almost) identical to the original ones.")
    dd = np.abs( rcA - pcaA.components_).flatten()

    __ = plt.hist(np.abs( rcA - pcaA.components_).flatten()  , bins=np.linspace( dd.min() ,dd.max() , 20)[1:] )
    plt.yscale("log")

else:
    print("Reconstructed PCAs look good.")


# In[237]:


# plot reconstructed - real pca/svd components
if True:
    f=plt.figure(figsize=[30,5])
    plt.imshow( rcA - pcaA.components_ )


# In[238]:


# the othogonality plot
if True:
    plotfn = "{}/pca_orthogA_{}_{}.pdf".format(dir_rebin, IFU,"".join(amps))
    f = plt.figure(figsize=[12,4])
    plt.subplot(121)
    plt.imshow(  np.log( np.abs(np.matmul(pcaA.components_, pcaA.components_.T ))), origin='bottom')
    plt.xlabel("comp. #")
    plt.ylabel("comp. #")
    plt.subplot(122)
    plt.imshow( np.log( np.abs(np.matmul(rcA, rcA.T )) ), origin='bottom')
    plt.xlabel("comp. #")
    plt.ylabel("comp. #")
    f.tight_layout()
    plt.savefig(plotfn)


# In[239]:


# save pca components of A
pca_comp_fname = "{}/pca_comp_A_{}_{}.pickle".format(dir_rebin, IFU, "".join(amps))
#pickle.dump(  (MA, pcaA.components_) , open(pca_comp_fname,'wb') , protocol=2   )
pickle.dump(  (MA, rcA) , open(pca_comp_fname,'wb') , protocol=2   )
print("Wrote {}".format(pca_comp_fname))


# In[240]:


#generic gaussian
def gauss(mu, sigma, x):
    return 1./(sigma * np.sqrt(2. * np.pi) ) * np.exp( -(x-mu)**2./(2. * sigma**2.))

gg = OrderedDict()

for amp in amps_skysub:
    wwcut, XBcut = build_XB(IFU, amp, ww, skys, wstart, wend)
    # now we subtract the mean of each column!, this is probably unnecassary as
    # the scikit learn PCA already does this, but it helps the plotting and so forth
    MB = np.mean(XBcut[jj],axis=0)
    if MEANSHIFT:
        XBmean = XBcut[jj] - MB
    else:
        XBmean = XBcut[jj]
        
        
    # Add synthetic lines in every B - spectrum
    # random wavelength
    # random sigma (3.5 - 10. A)
    # random amplitude (20 - 100. A)
    if TEST_MOCK_EMISSION:
        #g[ (shot, amp, fiber) ] = []

        for i in range(XBmean.shape[0]):
            for fiber in range(XBmean.shape[1]):
                mu = np.random.uniform()* (wwcut[-1] - wwcut[0]) + wwcut[0]
                sigma = np.random.uniform() * 10. + 3.5
                A = ( np.random.uniform() * 90. + 20.) 

                g = A*gauss(mu, sigma, wwcut)

                print("Adding fake emission line to: ", shotids[i][0], shotids[i][1], amp, fiber)
                XBcut[i,fiber]  = XBcut[i,fiber] + g

                gg[ (shotids[i][0], shotids[i][1], amp, fiber) ] = g
        
    ### reconstruct pca components of all B fibers through linear combination of spectra from B ###
    rcB = OrderedDict() # will hald for all fibers (in the current amp) all the pseudo PCA components
    for fiber in range(XBmean.shape[1]):
        # BUT using projection from A
        rcB[fiber] = np.matmul( XBmean[:,fiber,:].T, ccA).T
        # they wont be normalized yet
        for j,cB in enumerate(rcB[fiber]):
            rcB[fiber][j] = rcB[fiber][j]/np.linalg.norm(rcB[fiber][j])
            
    pca_comp_fname = "{}/pca_comp_B_{}_{}.pickle".format(dir_rebin,IFU,amp)
    pickle.dump(  (MB,rcB) , open(pca_comp_fname,'wb') , protocol=2   )
    print("Wrote {}".format(pca_comp_fname))
    
    if False:
        f = plt.figure()
        plt.imshow( np.arcsinh( np.matmul(rcB[75], rcB[75].T ) ), origin='bottom')


# In[241]:


if False:
    # compute projection of input spectra onto PCA basis
    #tA = pcaA.transform(XAmean)
    tA = np.matmul( XAmean, rcA.T)


    # make sure we can reconstuct the spectra from the
    # actual principal components but also from the reconstructed ones
    for i in range(10):
        f = plt.figure(figsize=[15,3])
        plt.subplot(131)
        plt.plot(wwcut, XAmean[i] )
        plt.plot(wwcut,  np.inner(tA, pcaA.components_.T)[i] )
        plt.plot(wwcut,  pcaA.inverse_transform(tA)[i] )


        plt.subplot(132)
        plt.plot(wwcut,  XAmean[i] )
        plt.plot(wwcut,  np.inner(tA, rcA.T)[i] )


        plt.subplot(133)
        plt.plot(wwcut,  np.inner(tA, pcaA.components_.T)[i] )
        plt.plot(wwcut,  np.inner(tA, rcA.T)[i] )


# # Now apply this to B

# In[242]:


# load pca components of A
pca_comp_fname = "{}/pca_comp_A_{}_{}.pickle".format(dir_rebin, IFU, "".join(amps))

print("Reading {}".format(pca_comp_fname))
MA, rcA = pickle.load( open(pca_comp_fname,'rb'), encoding='iso-8859-1' )

wws, skys = load_skys_for_shotlist(IFU, shotlist_skyrecon, amps)  

shotids = [k for k in skys[(IFU,amps[0])] ]

# Check that all wavelength arrays and spectra have same dimensions and sampling
# Here we need to pass previous sampling information
# to make sure that array sises are compatible.
ww, __, __, __, __, __ = homogenize(wws, skys, start0, stop0, length0, skyx0, skyy0)
    

# build data matrix A
wwcut, XAcut = build_XA(IFU, ww, skys, wstart, wend, amps)

# normalize 
# but normalize with respect to 
# the mean that we used fot the PCA component calculation
#MA = np.mean(XAcut,axis=0)
if MEANSHIFT:
    XAmean = XAcut - MA
else:
    XAmean = XAcut
# compute projection of input spectra onto PCA basis
ccA2 = np.inner(XAmean, rcA)


# In[243]:


from matplotlib.backends.backend_pdf import PdfPages

qa_pdf = "{}/pca_comp_B_{}.pdf".format(dir_rebin, IFU)

with PdfPages(qa_pdf) as pdf:

    for amp in amps_skysub:
        B = (IFU, amp) 

        # load pca components of B
        pca_comp_fname = "{}/pca_comp_B_{}_{}.pickle".format(dir_rebin, IFU,amp)
        MB, rcB = pickle.load( open(pca_comp_fname,'rb'), encoding='iso-8859-1' )

        
        wwcut, XBcut = build_XB(IFU, amp, ww, skys, wstart, wend)
        
        
        if TEST_MOCK_EMISSION:
            for i in range(XBmean.shape[0]):
                for fiber in range(XBmean.shape[1]):
                    try:
                        XBcut[i,fiber]  = XBcut[i,fiber] + gg[ (shotids[i][0], shotids[i][1], amp, fiber) ]
                    except:
                        #print("1 No info for ", (shotids[i][0], shotids[i][1], amp, fiber) )
                        pass
                
                
        # now we subtract the mean of each column!, this is probably unnecassary as
        # the scikit learn PCA already does this, but it helps the plotting and so forth
        #MB = np.mean(XBcut,axis=0)
        if MEANSHIFT:
            XBmean = XBcut - MB
        else:
            XBmean = XBcut

        ### reconstruct pca components of all B fibers through linear combination of spectra from B ###
        #rcB = OrderedDict() # will hald for all fibers (in the current amp) all the pseudo PCA components
        #for fiber in range(XBmean.shape[1]):
        #    # BUT using projection from A
        #    rcB[fiber] = np.matmul( XBmean[:,fiber,:].T, ccA2).T
        #    # they wont be normalized yet
        #    for j,cB in enumerate(rcB[fiber]):
        #        rcB[fiber][j] = rcB[fiber][j]/np.linalg.norm(rcB[fiber][j])

        # Now reconstruct sky for all exposures and fibers
        B_recon_sky = np.zeros_like(XBmean)
        for i in range(XBmean.shape[0]): # loop over exposures
            for fiber in range(XBmean.shape[1]):  # loop over fibers
                # now compute sky from pseudo PCA components of B
                # according to weights of A
                B_recon_sky[i,fiber,:] = np.inner(ccA2, rcB[fiber].T)[i] + MB[fiber,:]
                
                
                y  = XBmean[i,fiber,:] + MB[fiber,:] # original sky in B
                ry = B_recon_sky[i,fiber,:]          # reconstructed sky in B
                res = ry-y

                p = polyfit(wwcut, res, deg=5)
                B_recon_sky[i,fiber,:] -= polyval(p, wwcut)




        # Quality control
        # plot for 10 randomly picked exposures
        # recontrcuted sky and residuals of one (or a few) fiber(s)
        np.random.seed(42)
        qa_exposures = np.array( np.random.uniform(size=10) * XBmean.shape[0], dtype=int)
        qa_fibers = [75]

        from IPython.display import display
        for i in qa_exposures: # loop over exposures
            for fiber in qa_fibers:  # loop over fibers

                f = plt.figure(figsize=[15,3])
                ax = plt.subplot(121)

                y  = XAmean[i] + MA
                ry = np.inner(ccA2, rcA.T)[i] + MA
                res = ry-y
                plt.plot(wwcut,  y )
                plt.plot(wwcut,   ry )
                plt.twinx()
                plt.plot(wwcut,   res, 'g.'  )
                plt.ylim([-3.,10.])
                plt.text(0.9,0.9,"res.={:.3f}\n rel. res.={:.3f}".format(np.std(res), np.std(res)/np.abs(np.mean(y)) ), transform = ax.transAxes, ha='right',va='top')
                plt.text(0.05,0.95,"A\n{}\n{}\n{}".format(shotids[i][0], shotids[i][1], B[0],  fiber ), transform = ax.transAxes, ha='left',va='top')

                ax = plt.subplot(122, sharex=ax)
                y  = XBmean[i,fiber,:] + MB[fiber,:] # original sky in B
                ry = B_recon_sky[i,fiber,:]          # reconstructed sky in B
                res = y-ry

                plt.plot(wwcut,   y )
                plt.plot(wwcut,   ry )
                plt.twinx()

                plt.plot(wwcut,   res, 'g.' )
                plt.ylim([-3.,10.])
                plt.text(0.9,0.9,"res.={:.3f}\n rel. res.={:.3f}".format(np.std(res), np.std(res)/np.abs(np.mean(y)) ), transform = ax.transAxes, ha='right',va='top')
                plt.text(0.05,0.95,"B\n{}\n{}\n{}\n{}".format(shotids[i][0], shotids[i][1], B[0],  B[1], fiber ), transform = ax.transAxes, ha='left',va='top')
                
                if TEST_MOCK_EMISSION:
                    try:
                        plt.plot(wwcut,   gg[ (shotids[i][0], shotids[i][1], amp, fiber) ], 'r-' , drawstyle='steps-mid', alpha=.5)
                    except:
                        print("No info for ", (shotids[i][0], shotids[i][1], amp, fiber) )
                        pass
                    

                display(f)
                pdf.savefig()  # saves the current figure into a pdf page

                plt.close()


        for i,k in enumerate(shotids):

            save_sky(IFU, amp , k, wwcut, B_recon_sky[i], dir_rebin)



        #save_skys(B, pca_sky, pattern)


# In[244]:


if TEST_MOCK_EMISSION:
    # Finally, check how well we are doing:
    # Make sure we can reconstuct the spectra from the
    # actual principal components for A but also from the reconstructed ones for B
    fiber  = 75
    for i in range(5):
        f = plt.figure(figsize=[5,5])
        ax = plt.subplot()
        y  = XBmean[i,fiber,:] + MB[fiber,:] # original sky in B
        #ry = np.inner(tA, rcB.T)[i] + MB

        ry = np.inner(ccA2, rcB[fiber].T)[i] + MB[fiber,:]

        res = y-ry

        wc = wwcut[np.argmax( gg[ (shotids[i][0], shotids[i][1], amp, fiber) ])]
        ii = (wwcut > (wc-100.)) * (wwcut < (wc+100.))

        plt.plot(wwcut[ii],   res[ii], 'g-' , drawstyle='steps-mid')
        plt.ylim([-2,10.])
        plt.text(0.9,0.9,"res.={:.3f}\n rel. res.={:.3f}".format(np.std(res), np.std(res)/np.abs(np.mean(y)) ), transform = ax.transAxes, ha='right',va='top')

        #plt.plot(wwcut[ii],   gg[i][ii], 'r-' , drawstyle='steps-mid', alpha=.5)
        plt.plot(wwcut[ii],   gg[ (shotids[i][0], shotids[i][1], amp, fiber) ][ii], 'r-' , drawstyle='steps-mid', alpha=.5)

        plt.xlabel("wavelength [A]")


# In[245]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

qa_pdf = "{}/pca_vs_normal_B_{}.pdf".format(dir_rebin, IFU)

with PdfPages(qa_pdf) as pdf:

    for shot,exp in shotids[:]:
        for amp in amps_skysub:
            ff = glob.glob("{}/{}/{}/pca_multi_???_{}_???_{}_rebin.pickle".format(dir_rebin, shot,exp,IFU,amp))
            print(shot,exp,IFU,amp,":", ff)
            ww,dd = pickle.load( open(ff[0],'rb'), encoding='iso-8859-1' )

            vmin = -50.
            vmax =  50.

            #vmin = None
            #vmax = None
            f = plt.figure(figsize=[40,7])
            ax1 = plt.subplot(131)
            im1 = plt.imshow( dd["sky_subtracted"], vmin=vmin, vmax=vmax)
            plt.text(.1,1., "Classic sky subtraction    {} {} {} {}".format(shot,exp,IFU,amp), va='bottom', ha='left')
            
            ax2 = plt.subplot(132)
            im2 = plt.imshow( dd["pca_sky_subtracted"], vmin=vmin, vmax=vmax)
            plt.text(.1,1., "PCA sky subtraction", va='bottom', ha='left')

            ax3 = plt.subplot(133)
            im3 = plt.imshow( dd["sky_subtracted"]-(dd["pca_sky_subtracted"]), vmin=vmin/10., vmax=vmax/10.)
            plt.text(.1,1., "Difference", va='bottom', ha='left')

            for (ax,im) in [(ax1,im1), (ax2,im2), (ax3,im3)]:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
        
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='2%', pad=0.05)
                f.colorbar(im, cax=cax, orientation='vertical')
                #plt.show()
            display(f)
            pdf.savefig()  # saves the current figure into a pdf page

            plt.close()


# In[246]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

qa_pdf = "{}/pca_vs_normal_sky_B_{}.pdf".format(dir_rebin, IFU)

with PdfPages(qa_pdf) as pdf:

    for shot,exp in shotids[:]:
        for amp in amps_skysub:
            ff = glob.glob("{}/{}/{}/pca_multi_???_{}_???_{}_rebin.pickle".format(dir_rebin, shot,exp,IFU,amp))
            print(shot,exp,IFU,amp,":", ff)
            ww,dd = pickle.load( open(ff[0],'rb'), encoding='iso-8859-1' )

            vmin =  0.
            vmax =  400.

            #vmin = None
            #vmax = None
            f = plt.figure(figsize=[40,7])
            ax1 = plt.subplot(131)
            im1 = plt.imshow( dd["sky_spectrum"], vmin=vmin, vmax=vmax)
            plt.text(.1,1., "Classic sky spectrum    {} {} {} {}".format(shot,exp,IFU,amp), va='bottom', ha='left')
            
            ax2 = plt.subplot(132)
            im2 = plt.imshow( dd["pca_sky_spectrum"]+dd['sky_spectrum']-dd['pca_sky_spectrum'], vmin=vmin, vmax=vmax)
            plt.text(.1,1., "PCA sky spectrum", va='bottom', ha='left')

            ax3 = plt.subplot(133)
            im3 = plt.imshow( dd["sky_spectrum"]-dd["pca_sky_spectrum"], vmin=-10., vmax=10.)
            plt.text(.1,1., "Difference", va='bottom', ha='left')

            for (ax,im) in [(ax1,im1), (ax2,im2), (ax3,im3)]:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
        
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='2%', pad=0.05)
                f.colorbar(im, cax=cax, orientation='vertical')
                #plt.show()
            display(f)
            pdf.savefig()  # saves the current figure into a pdf page

            plt.close()


# In[ ]:




