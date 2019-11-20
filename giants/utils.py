from astropy import units as u
from astropy.constants import G
import numpy as np
import lightkurve as lk
from astropy.coordinates import SkyCoord, Angle
import matplotlib.pyplot as plt

def create_starry_model(time, rprs=.01, period=15., t0=5., i=90, ecc=0., m_star=1.):
    """ """
    import starry
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

def add_gaia_figure_elements(tpf, fig, magnitude_limit=18):
    """Make the Gaia Figure Elements"""
    # Get the positions of the Gaia sources
    c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        raise no_targets_found_message
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    coords = tpf.wcs.all_world2pix(radecs, 1) ## TODO, is origin supposed to be zero or one?
    year = ((tpf.astropy_time[0].jd - 2457206.375) * u.day).to(u.year)
    pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
    pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
    result.RA_ICRS += pmra
    result.DE_ICRS += pmdec

    # Gently size the points by their Gaia magnitude
    sizes = 10000.0 / 2**(result['Gmag']/2)

    plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
    plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='None', edgecolors='r', s=sizes)
    plt.xlim([tpf.column, tpf.column+tpf.shape[1]])
    plt.ylim([tpf.row, tpf.row+tpf.shape[2]])

    return fig
