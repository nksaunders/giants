import numpy as np
from astropy.stats import BoxLeastSquares
import lightkurve as lk
import astropy.units as u
from scipy.constants import G

def _calculate_separation(m_star, period):
    """ 
    Calculate the separation of a planet in a circular orbit around a star.
    
    Parameters
    ----------
    m_star : float
        Mass of the star in solar masses.
    period : float
        Orbital period of the planet in days.

    Returns
    -------
    a : float
        Semi-major axis of the planet in AU.
    """
    a = (((G*m_star*u.solMass/(4*np.pi**2))*(period*u.day)**2)**(1/3))
    return a.to(u.AU).value

def get_cutout(ticid, cutout_size=9):
    """ 
    Download a cutout of the TESS FFIs centered on a given TICID.
    
    Parameters
    ----------
    ticid : int
        TICID of the target.
    cutout_size : int
        Size of the cutout in pixels. Default is 9.

    Returns
    -------
    tpf : lightkurve.targetpixelfile.TessTargetPixelFile
        Cutout of the TESS FFIs centered on the target.
    """
    tpf = lk.search_tesscut(ticid)[0].download(cutout_size=cutout_size)

    return tpf

def build_ktransit_model(ticid, lc, period, t0, rprs=0.02, vary_transit=True):
    """
    Create a ktransit model for a given target and fit it to the light curve.

    Parameters
    ----------
    ticid : int
        TICID of the target.
    lc : lightkurve.lightcurve.TessLightCurve
        Light curve of the target.
    period : astropy.units.quantity.Quantity
        Orbital period of the planet.
    t0 : astropy.units.quantity.Quantity
        Time of first transit.
    rprs : float
        Radius of the planet in units of the stellar radius. Default is 0.02.
        
    Returns
    -------
    fitT : ktransit.fittransit.FitTransit
        ktransit model fit to the light curve.
    """
    from ktransit import FitTransit
    fitT = FitTransit()

    t0 = t0.value
    period = period.value

    fitT.add_guess_star(rho=0.022, zpt=0, ld1=0.6505,ld2=0.1041) 
    fitT.add_guess_planet(T0=t0, period=period, impact=0.5, rprs=rprs)

    ferr = np.ones_like(lc.time.value) * 0.00001
    fitT.add_data(time=lc.time.value,flux=lc.flux.value,ferr=ferr)

    vary_star = ['zpt']      # free stellar parameters
    if vary_transit:
        vary_planet = (['period', 'impact',       # free planetary parameters
            'T0', #'esinw', 'ecosw',
            'rprs']) #'impact',               # free planet parameters are the same for every planet you model
    else:
        vary_planet = (['rprs'])

    fitT.free_parameters(vary_star, vary_planet)
    fitT.do_fit()                   # run the fitting

    return fitT

def _individual_ktransit_dur(time, data):
    """ 
    Calculate the duration of a transit using ktransit.

    Parameters
    ----------
    time : astropy.units.quantity.Quantity
        Time array of the light curve.
    data : astropy.units.quantity.Quantity
        Flux array of the light curve.

    Returns
    -------
    dur : float
        Duration of the transit in hours.
    """
    inds = np.where(data < np.median(data))[0]
    first_transit = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)[0]

    dur = (time[first_transit[-1]] - time[first_transit[0]]) * 24.0

    return dur
