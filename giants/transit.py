import starry
from astropy import units as u
from astropy.constants import G
import numpy as np

def create_starry_model(time, rprs=.01, period=15., t0=5., i=90, ecc=0., m_star=1.):
    """ """
    # instantiate a starry primary object (star)
    star = starry.kepler.Primary()
    # calculate separation
    a = _calculate_separation(m_star, period)
    # quadradic limb darkening
    star[1] = 0.40
    star[2] = 0.26
    # instantiate a starry secondary object (planet)
    planet = starry.kepler.Secondary(lmax=5)
    # define its parameters
    planet.r = rprs * star.r
    planet.porb = period
    planet.tref = t0
    planet.inc = i
    planet.ecc = ecc
    planet.a = (a*u.AU).to(u.solRad).value # in units of stellar radius
    # create a system and compute its lightcurve
    system = starry.kepler.System(star, planet)
    system.compute(time)
    # return the light curve
    return system.lightcurve

def _calculate_separation(m_star, period):
    """ """
    a = (((G*m_star*u.solMass/(4*np.pi**2))*(period*u.day)**2)**(1/3))
    return a.to(u.AU).value
