import os
import re
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
from astropy.stats import BoxLeastSquares, mad_std, LombScargle
import astropy.stats as ass
from astropy.coordinates import SkyCoord, Angle
import lightkurve as lk
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u
import ktransit
import matplotlib.ticker as mtick
# import eleanor
try:
    from . import lomb
    from .utils import build_ktransit_model, _individual_ktransit_dur, find_ica_components, get_cutout
except:
    import lomb
    from utils import build_ktransit_model, _individual_ktransit_dur, find_ica_components, get_cutout


#optional imports
try:
    from sklearn.decomposition import FastICA
except:
    print('ICA not available.')

__all__ = ['plot_summary']

def make_ica_plot(tic, tpf=None):
    """
    """

    if tpf is None:
        tpf = lk.search_tesscut(f'TIC {tic}').download(cutout_size=11)
    raw_lc = tpf.to_lightcurve(aperture_mask='all')

    ##Perform ICA
    n_components = 20

    X = np.ascontiguousarray(np.nan_to_num(tpf.flux), np.float64)
    X_flat = X.reshape(len(tpf.flux), -1) #turns three dimensional into two dimensional

    f1 = np.reshape(X_flat, (len(X), -1))
    X_pix = f1 / np.nansum(X_flat, axis=-1)[:, None]

    ica = FastICA(n_components=n_components) #define n_components
    S_ = ica.fit_transform(X_pix)
    A_ = ica.mixing_ #combine x_flat to get x

    a = np.dot(S_.T, S_)
    a[np.diag_indices_from(a)] += 1e-5
    b = np.dot(S_.T, raw_lc.flux)

    w = np.linalg.solve(a, b)

    comp_lcs = []
    blss = []
    max_powers = []

    for i,s in enumerate(S_.T):
        component_lc = s * w[i]
        comp_lcs.append(component_lc)
        # plt.plot(component_lc + i*1e5)

        model = BoxLeastSquares(tpf.time, component_lc)
        results = model.autopower(0.16, minimum_period=.5, maximum_period=24.)
        # model = transitleastsquares(tpf.time, component_lc)
        # results = model.power()
        period, power = results.period, results.power
        blss.append([period, power])
        # print(results.depth_snr[np.argmax(power)])
        if (np.std(component_lc) > 1e4) or (np.abs(period[np.argmax(power)] - 14) < 2) or (results.depth[np.argmax(power)]/np.median(component_lc) < 0):
            power = [0]

        max_powers.append(np.max(power))

    # plt.ylim(-1e5, 10e5)

    best_pers = blss[np.argmax(max_powers)][0]
    best_powers = blss[np.argmax(max_powers)][1]

    period = best_pers[np.argmax(best_powers)]

    transit_lc = lk.LightCurve(time=tpf.time, flux=comp_lcs[np.argmax(max_powers)])

    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle(f'TIC {tic}')

    scale = np.median(raw_lc.flux) / 10

    for i,c in enumerate(comp_lcs):
        ax[0,0].plot(tpf.time, c + i*scale)
    ax[0,0].set_ylim(-scale, n_components*scale)
    ax[0,0].set_xlim(tpf.time[0], tpf.time[-1])
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Flux')
    ax[0,0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    ax[0,0].set_title('ICA Components')

    transit_lc.plot(ax=ax[0,1])
    ax[0,1].set_xlim(tpf.time[0], tpf.time[-1])
    ax[0,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    ax[0,1].set_title('ICA comp with max BLS power')

    transit_lc.remove_outliers(9).fold(period).scatter(ax=ax[0,2], c='k', label=f'Period={period:.2f}')
    transit_lc.remove_outliers(9).fold(period).bin(7).plot(ax=ax[0,2], c='r', lw=2, C='C1', label='Binned')
    ax[0,2].set_ylim(-5*np.std(transit_lc.flux), 2*np.std(transit_lc.flux))
    ax[0,2].set_xlim(-.5,.5)
    ax[0,2].set_title('Folded ICA Transit Component')

    A_useful = A_.reshape(11,11,n_components).T #reshape from 2d to 3d

    weighted_comp = A_useful[np.argmax(max_powers)].T * w[np.argmax(max_powers)]

    ax[1,0].imshow(weighted_comp, origin='lower')
    ax[1,1].imshow(tpf.flux[200], origin='lower')
    im = ax[1,2].imshow(weighted_comp / tpf.flux[200], origin='lower')

    ax[1,0].set_title('Weighted Transit Component')
    ax[1,1].set_title('TPF')
    ax[1,2].set_title('Model / Flux')

    plt.colorbar(im)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.patch.set_facecolor('white')
    fig.set_size_inches(10, 7)

    return fig

def plot_quicklook(lc, ticid, breakpoints, target_list, save_data=True, outdir=None):

    if outdir is None:
        outdir = os.path.join(self.PACKAGEDIR, 'outputs')

    time, flux, flux_err = lc.time, lc.flux, lc.flux_err

    model = BoxLeastSquares(time, flux)
    results = model.autopower(0.16, minimum_period=2., maximum_period=21.)
    period = results.period[np.argmax(results.power)]
    t0 = results.transit_time[np.argmax(results.power)]
    depth = results.depth[np.argmax(results.power)]
    depth_snr = results.depth_snr[np.argmax(results.power)]

    '''
    Plot Filtered Light Curve
    -------------------------
    '''
    plt.subplot2grid((4,4),(1,0),colspan=2)

    plt.plot(time, flux, 'k', label="filtered")
    for val in breakpoints:
        plt.axvline(val, c='b', linestyle='dashed')
    plt.legend()
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time')

    osample=5.
    nyq=283.

    # calculate FFT
    freq, amp, nout, jmax, prob = lomb.fasper(time, flux, osample, 3.)
    freq = 1000. * freq / 86.4
    bin = freq[1] - freq[0]
    fts = 2. * amp * np.var(flux * 1e6) / (np.sum(amp) * bin)

    use = np.where(freq < nyq + 150)
    freq = freq[use]
    fts = fts[use]

    # calculate ACF
    acf = np.correlate(fts, fts, 'same')
    freq_acf = np.linspace(-freq[-1], freq[-1], len(freq))

    fitT = build_ktransit_model(ticid=ticid, lc=lc, vary_transit=False)
    dur = _individual_ktransit_dur(fitT.time, fitT.transitmodel)

    freq = freq
    fts1 = fts/np.max(fts)
    fts2 = scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5)
    fts3 = scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50)

    '''
    Plot Periodogram
    ----------------
    '''
    plt.subplot2grid((4,4),(0,2),colspan=2,rowspan=4)
    plt.loglog(freq, fts/np.max(fts))
    plt.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5), color='C1', lw=2.5)
    plt.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50), color='r', lw=2.5)
    plt.axvline(283,-1,1, ls='--', color='k')
    plt.xlabel("Frequency [uHz]")
    plt.ylabel("Power")
    plt.xlim(10, 400)
    plt.ylim(1e-4, 1e0)

    # annotate with transit info
    font = {'family':'monospace', 'size':10}
    plt.text(10**1.04, 10**-3.50, f'depth = {depth:.4f}        ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    plt.text(10**1.04, 10**-3.62, f'depth_snr = {depth_snr:.4f}    ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    plt.text(10**1.04, 10**-3.74, f'period = {period:.3f} days    ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    plt.text(10**1.04, 10**-3.86, f't0 = {t0:.3f}            ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    try:
        # annotate with stellar params
        # won't work for TIC ID's not in the list
        if isinstance(ticid, str):
            ticid = int(re.search(r'\d+', str(ticid)).group())
        Gmag = target_list[target_list['ID'] == ticid]['GAIAmag'].values[0]
        Teff = target_list[target_list['ID'] == ticid]['Teff'].values[0]
        R = target_list[target_list['ID'] == ticid]['rad'].values[0]
        M = target_list[target_list['ID'] == ticid]['mass'].values[0]
        plt.text(10**1.7, 10**-3.50, rf"G mag = {Gmag:.3f} ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
        plt.text(10**1.7, 10**-3.62, rf"Teff = {int(Teff)} K  ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
        plt.text(10**1.7, 10**-3.74, rf"R = {R:.3f} $R_\odot$  ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
        plt.text(10**1.7, 10**-3.86, rf"M = {M:.3f} $M_\odot$    ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    except:
        pass

    # plot ACF inset
    ax = plt.gca()
    axins = inset_axes(ax, width=2.0, height=1.4)
    axins.plot(freq_acf, acf)
    axins.set_xlim(1,25)
    axins.set_xlabel("ACF [uHz]")

    '''
    Plot BLS
    --------
    '''
    plt.subplot2grid((4,4),(2,0),colspan=2)

    plt.plot(results.period, results.power, "k", lw=0.5)
    plt.xlim(results.period.min(), results.period.max())
    plt.xlabel("period [days]")
    plt.ylabel("log likelihood")

    # Highlight the harmonics of the peak period
    plt.axvline(period, alpha=0.4, lw=4)
    for n in range(2, 10):
        plt.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")

    phase = (t0 % period) / period
    foldedtimes = (((time - phase * period) / period) % 1)
    foldedtimes[foldedtimes > 0.5] -= 1
    foldtimesort = np.argsort(foldedtimes)
    foldfluxes = flux[foldtimesort]
    plt.subplot2grid((4,4), (3,0),colspan=2)
    plt.scatter(foldedtimes, flux, s=2)
    plt.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes, 40), lw=2, color='r', label=f'P={period:.2f} days, dur={dur:.2f} hrs')
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.0025, 0.0025)
    plt.legend(loc=0)

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.suptitle(f'{ticid}', fontsize=14)
    fig.set_size_inches(10, 7)

    if save_data:
        np.savetxt(outdir+'/timeseries/'+str(ticid)+'.dat.ts', np.transpose([time, flux]), fmt='%.8f', delimiter=' ')
        np.savetxt(outdir+'/fft/'+str(ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')
        with open(os.path.join(outdir,"transit_stats.txt"), "a+") as file:
            file.write(f"{ticid} {depth} {depth_snr} {period} {t0} {dur}\n")

    return fig

def plot_transit_vetting(ticid, period, t0, lc=None, tpf=None):
    """

    """
    if tpf is None:
        tpf = get_cutout(ticid)

    if lc is None:
        lc = tpf.to_lightcurve(aperture_mask='threshold')

    ica_lcs = find_ica_components(tpf)

    plt.clf()
    plt.figure(figsize=(12,12))
    fig = plt.subplot2grid((6,4),(0,0),colspan=2,rowspan=2)
    fig.patch.set_facecolor('white')

    tpf[100].plot(ax=fig, title='', show_colorbar=False)
    add_gaia_figure_elements(tpf, fig)

    fig = plt.subplot2grid((6,4),(2,0),colspan=2,rowspan=1)
    lc.fold(2*period, t0+period/2).scatter(ax=fig, c='k', label='Odd Transit')
    lc.fold(2*period, t0+period/2).bin(3).plot(ax=fig, c='C1', lw=2)
    plt.xlim(-.5, 0)
    rms = np.std(lc.flux)
    plt.ylim(-5*rms, 3*rms)

    fig = plt.subplot2grid((6,4),(3,0),colspan=2,rowspan=1)
    lc.fold(2*period, t0+period/2).scatter(ax=fig, c='k', label='Even Transit')
    lc.fold(2*period, t0+period/2).bin(3).plot(ax=fig, c='C1', lw=2)
    plt.xlim(0, .5)
    plt.ylim(-5*rms, 3*rms)

    fig = plt.subplot2grid((6,4),(0,2),colspan=4,rowspan=4)
    for i,lc in enumerate(ica_lcs):
        scale = .5
        plt.plot(lc + i*scale)
    plt.xlim(0, len(ica_lcs[0]))
    plt.ylim(-scale, len(ica_lcs)*scale)

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(10, 7)

    return fig

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
    coords = tpf.wcs.all_world2pix(radecs, 0) ## TODO, is origin supposed to be zero or one?
    year = ((tpf.time[0].jd - 2457206.375) * u.day).to(u.year)
    pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
    pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
    result.RA_ICRS += pmra
    result.DE_ICRS += pmdec

    # Gently size the points by their Gaia magnitude
    sizes = 10000.0 / 2**(result['Gmag']/2)

    plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
    plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='None', edgecolors='r', s=sizes)
    plt.xlim([tpf.column, tpf.column+tpf.shape[1]-1])
    plt.ylim([tpf.row, tpf.row+tpf.shape[2]-1])
    plt.axis('off')

    return fig

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

def plot_summary(target, outdir=None, save_data=False, save_fig=True):
    """

    """
    print(f"Generating plot summary for TIC {target.ticid}.\n"
          "This may take a few minutes.")
    if outdir is None:
        outdir = os.path.join(target.PACKAGEDIR, 'outputs')

    font = {'family' : 'sans',
        'size'   : 11}
    matplotlib.rc('font', **font)
    plt.style.use('seaborn-muted')

    dims=(20, 12)

    ax = plt.subplot2grid(dims, (0,0), colspan=12, rowspan=3)
    plot_raw_lc(target, ax)
    param_string = stellar_params(target)
    plt.title(f'TIC {target.ticid}, ' + param_string)

    ax = plt.subplot2grid(dims, (4,6), colspan=6, rowspan=2)
    bls_results = get_bls_results(target.lc)
    period = bls_results.period[np.argmax(bls_results.power)]
    t0 = bls_results.transit_time[np.argmax(bls_results.power)]
    depth = bls_results.depth[np.argmax(bls_results.power)]
    depth_snr = bls_results.depth_snr[np.argmax(bls_results.power)]
    plot_bls(target.lc, ax, results=bls_results)
    plt.subplots_adjust(hspace=0)

    ax = plt.subplot2grid(dims, (6,6), colspan=6, rowspan=6)
    freq, fts = plot_fft(target.lc, ax)

    ax = plt.subplot2grid(dims, (4,0), colspan=5, rowspan=2)
    plot_folded(target.lc, period.value, t0.value, depth, ax)

    ax = plt.subplot2grid(dims, (7,0), colspan=5, rowspan=5)
    plot_tpf(target, ax)

    ax = plt.subplot2grid(dims, (13,0), colspan=5, rowspan=2)
    plot_even(target.lc, period.value, t0.value, depth, ax)

    ax = plt.subplot2grid(dims, (15,0), colspan=5, rowspan=2)
    plot_odd(target.lc, period.value, t0.value, depth, ax)
    plt.subplots_adjust(hspace=0)

    ax = plt.subplot2grid(dims, (13,6), colspan=6, rowspan=3)
    model_lc, ktransit_model = fit_transit_model(target)
    result = ktransit_model.fitresult[1:]
    kt_period = result[0]
    kt_t0 = result[2]
    dur = _individual_ktransit_dur(model_lc.time, model_lc.flux)
    plot_tr_top(target.lc, model_lc, kt_period, kt_t0, ax)

    ax = plt.subplot2grid(dims, (16,6), colspan=6, rowspan=1)
    plot_tr_bottom(target.lc, model_lc, kt_period, kt_t0, ax)
    plt.subplots_adjust(hspace=0)

    ax = plt.subplot2grid(dims, (19,0), colspan=12, rowspan=1)
    plot_table(target, model_lc, ktransit_model, depth_snr, dur, ax)

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(dims[::-1])

    if save_data:
        np.savetxt(outdir+'/timeseries/'+str(target.ticid)+'.dat.ts', np.transpose([target.lc.time, target.lc.flux]), fmt='%.8f', delimiter=' ')
        np.savetxt(outdir+'/fft/'+str(target.ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')
        with open(os.path.join(outdir, "transit_stats.txt"), "a+") as file:
            file.write(f"{target.ticid} {depth} {depth_snr} {period} {t0} {dur}\n")

    if save_fig:
        fig.savefig(str(outdir)+'/plots/'+str(target.ticid)+'_summary.png')

def fit_transit_model(target):
    """

    """

    ktransit_model = build_ktransit_model(target.ticid, target.lc)

    model_lc = lk.LightCurve(time=target.lc.time, flux=ktransit_model.transitmodel)
    return model_lc, ktransit_model

def plot_raw_lc(target, ax=None):
    """
    """
    if ax is None:
        _, ax = plt.subplots(1)


    target.lc.plot(ax=ax, c='k')
    ax.set_xlim(target.lc.time.value[0], target.lc.time.value[-1])
    for b in target.breakpoints:
        ax.axvline(b.value, linestyle='--', color='r')

def plot_tr_top(flux_lc, model_lc, per, t0, ax):
    res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
    res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)

    depth = 0 - np.min(model_lc.flux.value)

    ax.set_xticklabels([])
    ax.set_xlim(-.1*per, .1*per)
    ax.set_ylim(np.min(model_lc.flux.value)-depth*2, depth*2)

    flux_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=50)
    flux_lc.fold(per, t0).remove_outliers().bin(.1).scatter(ax=ax, c='r', s=300, alpha=.75)
    model_lc.fold(per, t0).plot(ax=ax, c='k', lw=2)

    ax.set_ylabel('Normalized Flux')

def plot_tr_bottom(flux_lc, model_lc, per, t0, ax):
    res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
    res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)
    res_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=50)
    res_lc.fold(per, t0).remove_outliers().bin(.1).scatter(ax=ax, c='r', s=300, alpha=.75)
    ax.axhline(0, c='k', linestyle='dashed')
    ax.set_xlim(-.1*per, .1*per)
    ax.set_ylabel('Residuals (ppm)')

def plot_fft(lc, ax=None):
    if ax is None:
        _, ax = plt.subplots(1)

    osample=5.
    nyq=283.

    time = lc.time
    flux = lc.flux

    # calculate FFT
    freq, amp, nout, jmax, prob = lomb.fasper(time, flux, osample, 3.)
    freq = 1000. * freq / 86.4
    bin = freq[1] - freq[0]
    fts = 2. * amp * np.var(flux * 1e6) / (np.sum(amp) * bin)

    use = np.where(freq.value < nyq + 150)
    freq = freq[use]
    fts = fts[use]

    ax.loglog(freq, fts/np.max(fts))
    ax.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5), color='C1', lw=2.5)
    ax.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50), color='r', lw=2.5)
    ax.axvline(283,-1,1, ls='--', color='k')
    ax.set_xlabel("Frequency [uHz]")
    ax.set_ylabel("Power")
    ax.set_xlim(10, 400)
    ax.set_ylim(1e-4, 1e0)

    return freq, fts

def get_bls_results(lc):
    model = BoxLeastSquares(lc.time, lc.flux)
    results = model.autopower(0.16, minimum_period=1., maximum_period=21.)
    return results

def plot_bls(lc, ax, results=None):

    time, flux, flux_err = lc.time, lc.flux, lc.flux_err

    if results is None:
        results = get_bls_results(lc)
    period = results.period[np.argmax(results.power)]
    t0 = results.transit_time[np.argmax(results.power)]
    depth = results.depth[np.argmax(results.power)]
    depth_snr = results.depth_snr[np.argmax(results.power)]

    ax.plot(results.period, results.power, "k", lw=0.5)
    ax.set_xlim(results.period.min().value, results.period.max().value)
    ax.set_xlabel("period [days]")
    ax.set_ylabel("log likelihood")

    # Highlight the harmonics of the peak period
    ax.axvline(period.value, alpha=0.4, lw=4)
    for n in range(2, 10):
        ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")

def plot_folded(lc, period, t0, depth, ax):

    if ax is None:
        _, ax = plt.subplots(1)

    time, flux, flux_err = lc.time, lc.flux, lc.flux_err

    phase = (t0 % period) / period
    foldedtimes = (((time.value - phase * period) / period) % 1)
    foldedtimes[foldedtimes > 0.5] -= 1
    foldtimesort = np.argsort(foldedtimes)
    foldfluxes = flux[foldtimesort]

    ax.plot(foldedtimes, flux, 'k.', markersize=2)
    ax.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes, 40), lw=2, color='r')#, label=f'P={period:.2f} days, dur={dur:.2f} hrs')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Flux')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-3*depth, 2*depth)
    plt.grid(True)

def plot_odd(lc, period, t0, depth, ax):

    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='k', label='Odd Transit')
    lc.fold(2*period, t0+period/2).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(0, .5*period)
    ax.set_ylim(-3*depth, 2*depth)

    plt.grid(True)

def plot_even(lc, period, t0, depth, ax):

    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='k', label='Even Transit')
    lc.fold(2*period, t0+period/2).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(-.5*period, 0)
    ax.set_ylim(-3*depth, 2*depth)

    plt.grid(True)

def plot_tpf(target, ax):
    ax = target.tpf.plot(ax=ax, show_colorbar=False, frame=100)
    ax = add_gaia_figure_elements(target.tpf, ax)

def plot_table(target, model_lc, ktransit_model, depth_snr, dur, ax):
    result = ktransit_model.fitresult[1:]

    col_labels = ['Period (days)', 'b', 't0', 'Rp/Rs', 'Duration (hours)', 'Depth SNR']
    values = [f'{val:.3f}' for val in result]
    values.append(f'{dur.value:.3f}')
    values.append(f'{depth_snr:.3f}')

    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=[values], colLabels=col_labels, loc='center')

def stellar_params(target):
    # from astroquery.mast import Catalogs
    # catalog_data = Catalogs.query_criteria(objectname=f'TIC {target.ticid}', catalog="Tic", radius=.0001, Bmag=[0,20])
    #
    # ra = catalog_data['ra'][0]
    # dec = catalog_data['dec'][0]
    # coords = f'({ra:.2f}, {dec:.2f})'
    # rstar = catalog_data['rad'][0]
    # teff = catalog_data['Teff'][0]
    # if np.isnan(rstar):
    #     rstar = '?'
    # else:
    #     rstar = f'{rstar:.2f}'
    # if np.isnan(teff):
    #     teff = '?'
    # else:
    #     teff = f'{teff:.0f}'
    # logg = catalog_data['logg'][0]
    # if np.isnan(logg):
    #     logg = '?'
    # else:
    #     logg = f'{logg:.2f}'
    # V = catalog_data['Vmag'][0]

    coords = f'({float(target.ra):.2f}, {float(target.dec):.2f})'
    rstar = float(target.target_row['rad'].values[0])
    teff = float(target.target_row['Teff'].values[0])
    if np.isnan(rstar):
        rstar = '?'
    else:
        rstar = f'{rstar:.2f}'
    if np.isnan(teff):
        teff = '?'
    else:
        teff = f'{teff:.0f}'
    logg = float(target.target_row['logg'].values[0])
    if np.isnan(logg):
        logg = '?'
    else:
        logg = f'{logg:.2f}'
    V = target.target_row['Vmag'].values[0]

    param_string = rf'(RA, dec)={coords}, R_star={rstar} $R_\odot$, logg={logg}, Teff={teff} K, V={float(V):.2f}'

    return param_string

def superplot(lc, ticid, breakpoints, target_list, save_data=False, outdir=None):
    """

    """

    time, flux, flux_err = lc.time, lc.flux, lc.flux_err

    model = BoxLeastSquares(time, flux)
    results = model.autopower(0.16, minimum_period=2., maximum_period=21.)
    period = results.period[np.argmax(results.power)]
    t0 = results.transit_time[np.argmax(results.power)]
    depth = results.depth[np.argmax(results.power)]
    depth_snr = results.depth_snr[np.argmax(results.power)]

    '''
    Plot Filtered Light Curve
    -------------------------
    '''
    plt.subplot2grid((8,16),(1,0),colspan=4, rowspan=1)

    plt.plot(time, flux, 'k', label="filtered")
    for val in breakpoints:
        plt.axvline(val, c='b', linestyle='dashed')
    plt.legend()
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time')

    osample=5.
    nyq=283.

    # calculate FFT
    freq, amp, nout, jmax, prob = lomb.fasper(time, flux, osample, 3.)
    freq = 1000. * freq / 86.4
    bin = freq[1] - freq[0]
    fts = 2. * amp * np.var(flux * 1e6) / (np.sum(amp) * bin)

    use = np.where(freq < nyq + 150)
    freq = freq[use]
    fts = fts[use]

    # calculate ACF
    acf = np.correlate(fts, fts, 'same')
    freq_acf = np.linspace(-freq[-1], freq[-1], len(freq))

    fitT = build_ktransit_model(ticid=ticid, lc=lc, vary_transit=False)
    dur = _individual_ktransit_dur(fitT.time, fitT.transitmodel)

    freq = freq
    fts1 = fts/np.max(fts)
    fts2 = scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5)
    fts3 = scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50)

    '''
    Plot Periodogram
    ----------------
    '''
    plt.subplot2grid((8,16),(0,4),colspan=4,rowspan=4)
    plt.loglog(freq, fts/np.max(fts))
    plt.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5), color='C1', lw=2.5)
    plt.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50), color='r', lw=2.5)
    plt.axvline(283,-1,1, ls='--', color='k')
    plt.xlabel("Frequency [uHz]")
    plt.ylabel("Power")
    plt.xlim(10, 400)
    plt.ylim(1e-4, 1e0)

    # annotate with transit info
    font = {'family':'monospace', 'size':10}
    plt.text(10**1.04, 10**-3.50, f'depth = {depth:.4f}        ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    plt.text(10**1.04, 10**-3.62, f'depth_snr = {depth_snr:.4f}    ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    plt.text(10**1.04, 10**-3.74, f'period = {period:.3f} days    ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    plt.text(10**1.04, 10**-3.86, f't0 = {t0:.3f}            ', fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    try:
        # annotate with stellar params
        # won't work for TIC ID's not in the list
        if isinstance(ticid, str):
            ticid = int(re.search(r'\d+', str(ticid)).group())
        Gmag = target_list[target_list['ID'] == ticid]['GAIAmag'].values[0]
        Teff = target_list[target_list['ID'] == ticid]['Teff'].values[0]
        R = target_list[target_list['ID'] == ticid]['rad'].values[0]
        M = target_list[target_list['ID'] == ticid]['mass'].values[0]
        plt.text(10**1.7, 10**-3.50, rf"G mag = {Gmag:.3f} ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
        plt.text(10**1.7, 10**-3.62, rf"Teff = {int(Teff)} K  ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
        plt.text(10**1.7, 10**-3.74, rf"R = {R:.3f} $R_\odot$  ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
        plt.text(10**1.7, 10**-3.86, rf"M = {M:.3f} $M_\odot$    ", fontdict=font).set_bbox(dict(facecolor='white', alpha=.9, edgecolor='none'))
    except:
        pass

    '''# plot ACF inset
    ax = plt.gca()
    axins = inset_axes(ax, width=2.0, height=1.4)
    axins.plot(freq_acf, acf)
    axins.set_xlim(1,25)
    axins.set_xlabel("ACF [uHz]")'''

    '''
    Plot BLS
    --------
    '''
    plt.subplot2grid((8,16),(2,0),colspan=4, rowspan=1)

    plt.plot(results.period, results.power, "k", lw=0.5)
    plt.xlim(results.period.min(), results.period.max())
    plt.xlabel("period [days]")
    plt.ylabel("log likelihood")

    # Highlight the harmonics of the peak period
    plt.axvline(period, alpha=0.4, lw=4)
    for n in range(2, 10):
        plt.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")

    phase = (t0 % period) / period
    foldedtimes = (((time - phase * period) / period) % 1)
    foldedtimes[foldedtimes > 0.5] -= 1
    foldtimesort = np.argsort(foldedtimes)
    foldfluxes = flux[foldtimesort]
    plt.subplot2grid((8,16), (3,0),colspan=2)
    plt.scatter(foldedtimes, flux, s=2)
    plt.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes, 40), lw=2, color='r', label=f'P={period:.2f} days, dur={dur:.2f} hrs')
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.0025, 0.0025)
    plt.legend(loc=0)

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.suptitle(f'{ticid}', fontsize=14)
    fig.set_size_inches(12, 10)

    if save_data:
        np.savetxt(outdir+'/timeseries/'+str(ticid)+'.dat.ts', np.transpose([time, flux]), fmt='%.8f', delimiter=' ')
        np.savetxt(outdir+'/fft/'+str(ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')
        with open(os.path.join(outdir,"transit_stats.txt"), "a+") as file:
            file.write(f"{ticid} {depth} {depth_snr} {period} {t0} {dur}\n")

    """
    ---------------
    TRANSIT VETTING
    ---------------
    """
    tpf = get_cutout(ticid, cutout_size=11)
    ica_lcs = find_ica_components(tpf)

    fig = plt.subplot2grid((8,16),(0,8),colspan=4,rowspan=4)
    fig.patch.set_facecolor('white')

    tpf.plot(ax=fig, title='', show_colorbar=False)
    add_gaia_figure_elements(tpf, fig)

    fig = plt.subplot2grid((8,16),(2,8),colspan=4,rowspan=2)
    lc.fold(2*period, t0+period/2).scatter(ax=fig, c='k', label='Odd Transit')
    lc.fold(2*period, t0+period/2).bin(3).plot(ax=fig, c='C1', lw=2)
    plt.xlim(-.5, 0)
    rms = np.std(lc.flux)
    plt.ylim(-3*rms, rms)

    fig = plt.subplot2grid((8,16),(3,8),colspan=4,rowspan=2)
    lc.fold(2*period, t0+period/2).scatter(ax=fig, c='k', label='Even Transit')
    lc.fold(2*period, t0+period/2).bin(3).plot(ax=fig, c='C1', lw=2)
    plt.xlim(0, .5)
    plt.ylim(-3*rms, rms)

    fig = plt.subplot2grid((8,16),(0,12),colspan=4,rowspan=4)
    for i,ilc in enumerate(ica_lcs):
        scale = 1
        plt.plot(ilc + i*scale)
    plt.xlim(0, len(ica_lcs[0]))
    plt.ylim(-scale, len(ica_lcs)*scale)

    """
    STARRY MODEL
    ------------
    """
    from .utils import _fit

    x, y, yerr = lc.time, lc.flux, lc.flux_err
    model, static_lc = _fit(x, y, yerr, target_list=target_list)

    model_lc = lk.LightCurve(time=x, flux=static_lc)

    with model:
        period = model.map_soln['period'][0]
        t0 = model.map_soln['t0'][0]
        r_pl = model.map_soln['r_pl'] * 9.96
        a = model.map_soln['a'][0]
        b = model.map_soln['b'][0]

    try:
        r_star = target_list[target_list['ID'] == ticid]['rad'].values[0]
    except:
        r_star = 10.

    fig = plt.subplot2grid((8,16),(4,0),colspan=4,rowspan=2)
    '''
    Plot unfolded transit
    ---------------------
    '''
    lc.scatter(c='k', label='Corrected Flux')
    lc.bin(binsize=7).plot(c='b', lw=1.5, alpha=.75, label='binned')
    model_lc.plot(c='r', lw=2, label='Transit Model')
    plt.ylim([-.002, .002])
    plt.xlim([lc.time[0], lc.time[-1]])

    fig = plt.subplot2grid((8,16),(6,0),colspan=4,rowspan=2)
    '''
    Plot folded transit
    -------------------
    '''
    lc.fold(period, t0).scatter(c='k', label=rf'$P={period:.3f}, t0={t0:.3f}, '
                                                         'R_p={r_pl:.3f} R_J, b={b:.3f}')
    lc.fold(period, t0).bin(binsize=7).plot(c='b', alpha=.75, lw=2)
    model_lc.fold(period, t0).plot(c='r', lw=2)
    plt.xlim([-0.5, .5])
    plt.ylim([-.002, .002])
