
import sys
import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from sklearn.decomposition import PCA
import pickle
import os
import glob
from matplotlib import pyplot as plt
from collections import OrderedDict
import spectrum
from numpy import polyfit,polyval


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
    wwcut = ww[ii]

    return wwcut, XBcut


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
    
    ww, starts, stops, lenghts, skyx, skyy, same_starts, same_stops, same_lenghts, same_skyx, same_skyy = \
        check(wws, skys, start0, stop0, length0, skyx0, skyy0)
    
    
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
        ww, starts, stops, lenghts, skyx, skyy, same_starts, same_stops, same_lenghts, same_skyx, same_skyy = \
            check(wws, skys, start0, stop0, length0, skyx0, skyy0)

    
    # if all good return wavelength array that is good for all
    if same_starts and same_stops and same_lenghts and same_skyx:
            
        return ww, starts[0], stops[0], lenghts[0], skyx[0],  skyy[0]
    else:
        print("All are not equal. Stopping here.")
        sys.exit(1)

        
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
        
        
           
def load_skys_for_shotlist(dir_rebin, IFU, shotlist, amps):
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
                fff = glob.glob(pattern)
                if len(fff) == 0:
                    print("No file found like {}. Check rebin dir.".format(pattern))
                    continue
                amp_files[amp] = fff[0]

            amp_has_data = [  amp in amp_files.keys()  for amp in amps]
            if all(amp_has_data):
                for amp in amps:
                    ff[amp] += [amp_files[amp]]
            else:
                print("WARNING: for {} exp {}, not all requested amps have data, dropping ....".format(shot, e))
    skys = OrderedDict()
    wws  = OrderedDict()
    
    for amp in amps:
        wws[(IFU,amp)],skys[(IFU,amp)] = load_skys(ff[amp],which="sky_spectrum")

    return wws,skys
