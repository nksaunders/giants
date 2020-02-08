import os
import re
import sys
import eleanor
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares, mad_std, LombScargle
import astropy.stats as ass
import lightkurve as lk
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import FastICA
import astropy.units as u
import ktransit
try:
    from . import lomb
except:
    import lomb


def _calculate_separation(m_star, period):
    """ """
    a = (((G*m_star*u.solMass/(4*np.pi**2))*(period*u.day)**2)**(1/3))
    return a.to(u.AU).value

def _fit(x, y, yerr, ticid=None, target_list=None, period_prior=None, t0_prior=None, depth=None, **kwargs):
    """A helper function to generate a PyMC3 model and optimize parameters.

    Parameters
    ----------
    x : array-like
        The time series in days
    y : array-like
        The light curve flux values
    yerr : array-like
        Errors on the flux values
    """

    try:
        import pymc3 as pm
        import theano.tensor as tt
        import exoplanet as xo
    except:
        raise(ImportError)

    def build_model(x, y, yerr, period_prior, t0_prior, depth, minimum_period=2, maximum_period=30, r_star_prior=5.0, t_star_prior=5000, m_star_prior=None, start=None):
        """Build an exoplanet model for a dataset and set of planets

        Paramters
        ---------
        x : array-like
            The time series (in days); this should probably be centered
        y : array-like
            The relative fluxes (in parts per thousand)
        yerr : array-like
            The uncertainties on ``y``
        period_prior : list
            The literature values for periods of the planets (in days)
        t0_prior : list
            The literature values for phases of the planets in the same
            coordinates as `x`
        rprs_prior : list
            The literature values for the ratio of planet radius to star
            radius
        start : dict
            A dictionary of model parameters where the optimization
            should be initialized

        Returns:
            A PyMC3 model specifying the probabilistic model for the light curve

        """

        model = BoxLeastSquares(x, y)
        results = model.autopower(0.16, minimum_period=minimum_period, maximum_period=maximum_period)
        if period_prior is None:
            period_prior = results.period[np.argmax(results.power)]
        if t0_prior is None:
            t0_prior = results.transit_time[np.argmax(results.power)]
        if depth is None:
            depth = results.depth[np.argmax(results.power)]

        period_prior = np.atleast_1d(period_prior)
        t0_prior = np.atleast_1d(t0_prior)
        # rprs_prior = np.atleast_1d(rprs_prior)

        with pm.Model() as model:

            # Set model variables
            model.x = np.asarray(x, dtype=np.float64)
            model.y = np.asarray(y, dtype=np.float64)
            model.yerr = np.asarray(yerr + np.zeros_like(x), dtype=np.float64)

            '''Stellar Parameters'''
            # The baseline (out-of-transit) flux for the star in ppt
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            try:
                r_star_mu = target_list[target_list['ID'] == ticid]['rad'].values[0]
            except:
                r_star_mu = r_star_prior
            if m_star_prior is None:
                try:
                    m_star_mu = target_list[target_list['ID'] == ticid]['mass'].values[0]
                except:
                    m_star_mu = 1.2
                if np.isnan(m_star_mu):
                    m_star_mu = 1.2
            else:
                m_star_mu = m_star_prior
            r_star = pm.Normal("r_star", mu=r_star_mu, sd=1.)
            m_star = pm.Normal("m_star", mu=m_star_mu, sd=1.)
            t_star = pm.Normal("t_star", mu=t_star_prior, sd=200)
            rho_star_mu = ((m_star_mu*u.solMass).to(u.g) / ((4/3) * np.pi * ((r_star_mu*u.solRad).to(u.cm))**3)).value
            rho_star = pm.Normal("rho_star", mu=rho_star_mu, sd=.25)

            '''Orbital Parameters'''
            # The time of a reference transit for each planet
            t0 = pm.Normal("t0", mu=t0_prior, sd=2., shape=1)
            period = pm.Uniform("period", testval=period_prior,
                                lower=minimum_period,
                                upper=maximum_period,
                                shape=1)

            b = pm.Uniform("b", testval=0.5, shape=1)

            # Set up a Keplerian orbit for the planets
            model.orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, r_star=r_star, m_star=m_star)#rho_star=rho_star)

            # track additional orbital parameters
            a = pm.Deterministic("a", model.orbit.a)
            incl = pm.Deterministic("incl", model.orbit.incl)

            '''Planet Parameters'''
            # quadratic limb darkening paramters
            u_ld = xo.distributions.QuadLimbDark("u_ld")

            estimated_rpl = r_star*(depth)**(1/2)

            # logr = pm.Normal("logr", testval=np.log(estimated_rpl), sd=1.)
            r_pl = pm.Uniform("r_pl",
                              testval=estimated_rpl,
                              lower=0.,
                              upper=1.)

            # r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            rprs = pm.Deterministic("rprs", r_pl / r_star)
            teff = pm.Deterministic('teff', t_star * tt.sqrt(0.5*(1/a)))

            # Compute the model light curve using starry
            model.light_curves = xo.StarryLightCurve(u_ld).get_light_curve(
                                    orbit=model.orbit, r=r_pl, t=model.x)

            model.light_curve = pm.math.sum(model.light_curves, axis=-1) + mean


            pm.Normal("obs",
                      mu=model.light_curve,
                      sd=model.yerr,
                      observed=model.y)

            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point
            map_soln = xo.optimize(start=start, vars=[period, t0])
            map_soln = xo.optimize(start=map_soln, vars=[r_pl, mean])
            map_soln = xo.optimize(start=map_soln, vars=[period, t0, mean])
            map_soln = xo.optimize(start=map_soln, vars=[r_pl, mean])
            map_soln = xo.optimize(start=map_soln)
            model.map_soln = map_soln

        return model

    # build our initial model and store a static version of the output for plotting
    model = build_model(x, y, yerr, period_prior, t0_prior, depth, **kwargs)
    with model:
        mean = model.map_soln["mean"]
        static_lc = xo.utils.eval_in_model(model.light_curves, model.map_soln)

    return model, static_lc

def get_cutout(ticid, cutout_size=9):
    """ """
    tpf = lk.search_tesscut(ticid)[0].download(cutout_size=cutout_size)

    return tpf

def find_ica_components(tpf):
    """ """
    ##Perform ICA
    n_components = 20

    raw_lc = tpf.to_lightcurve(aperture_mask='all')

    X = np.ascontiguousarray(np.nan_to_num(tpf.flux), np.float64)
    X_flat = X.reshape(len(tpf.flux), -1) #turns three dimensional into two dimensional

    f1 = np.reshape(X_flat, (len(X), -1))
    X_pix = f1 / np.nansum(X_flat, axis=-1)[:, None]

    ica = FastICA(n_components=n_components) #define n_components
    S_ = ica.fit_transform(X_pix)
    A_ = ica.mixing_ #combine x_flat to get x

    # solve weights
    a = np.dot(S_.T, S_)
    a[np.diag_indices_from(a)] += 1e-5
    b = np.dot(S_.T, raw_lc.flux)

    w = np.linalg.solve(a, b)

    # normalize to get sign of weight
    w = [weight / np.abs(weight) for weight in w]

    comp_lcs = []
    for i,s in enumerate(S_.T):
        component_lc = s * w[i]
        comp_lcs.append(component_lc)

    return comp_lcs

def build_ktransit_model(ticid, lc, rprs=0.02, vary_transit=True):
    from ktransit import FitTransit
    fitT = FitTransit()

    model = BoxLeastSquares(lc.time, lc.flux)
    results = model.autopower(0.16, minimum_period=2., maximum_period=21.)
    period = results.period[np.argmax(results.power)]
    t0 = results.transit_time[np.argmax(results.power)]
    if rprs is None:
        depth = results.depth[np.argmax(results.power)]
        rprs = depth ** 2

    fitT.add_guess_star(rho=0.022, zpt=0, ld1=0.6505,ld2=0.1041) #come up with better way to estimate this using AS
    fitT.add_guess_planet(T0=t0, period=period, impact=0.5, rprs=rprs)

    ferr = np.ones_like(lc.time) * 0.00001
    fitT.add_data(time=lc.time,flux=lc.flux,ferr=ferr)#*1e-3)

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
    """ """
    inds = np.where(data < np.median(data))[0]
    first_transit = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)[0]

    dur = (time[first_transit[-1]] - time[first_transit[0]]) * 24.0

    return dur
