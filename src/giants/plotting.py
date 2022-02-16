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
    sizes = 50000.0 / 2**(result['Gmag']/2)

    plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
    plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='None', edgecolors='r', s=sizes)
    plt.xlim([tpf.column, tpf.column+tpf.shape[1]-1])
    plt.ylim([tpf.row, tpf.row+tpf.shape[2]-1])
    plt.axis('off')

    return fig

def plot_summary(target, outdir='', save_data=False, save_fig=True):
    """

    """

    font = {'family' : 'sans',
        'size'   : 14}
    matplotlib.rc('font', **font)
    plt.style.use('seaborn-muted')

    dims=(18, 24)

    # generate ktransit fit
    model_lc, ktransit_model = fit_transit_model(target)
    result = ktransit_model.fitresult[1:]
    kt_period = result[0]
    kt_t0 = result[2]
    dur = _individual_ktransit_dur(model_lc.time, model_lc.flux)

    # fit BLS
    bls_results = get_bls_results(target.lc)
    period = bls_results.period[np.argmax(bls_results.power)]
    t0 = bls_results.transit_time[np.argmax(bls_results.power)]
    depth = bls_results.depth[np.argmax(bls_results.power)]
    depth_snr = depth / np.std(target.lc.flux.value)
    # depth_snr = bls_results.depth_snr[np.argmax(bls_results.power)]

    scaled_residuals = np.median(fit_transit_model(target)[1].residuals()) / np.std(target.lc.flux.value)

    fig = plt.gcf()
    fig.suptitle(f'TIC {target.ticid}', fontweight='bold', size=24, y=0.93)

    ax = plt.subplot2grid(dims, (0,0), colspan=24, rowspan=3)
    plot_raw_lc(target, ax)
    param_string = stellar_params(target)
    ax.set_title(param_string, size=20)

    ax = plt.subplot2grid(dims, (4,0), colspan=16, rowspan=3)
    plot_folded(target.lc, period.value, t0.value, depth, ax)

    ax = plt.subplot2grid(dims, (4,17), colspan=7, rowspan=7)
    plot_tpf(target, ax)

    ax = plt.subplot2grid(dims, (8,0), colspan=8, rowspan=3)
    plot_even(target.lc, period.value, t0.value, depth, ax)

    ax = plt.subplot2grid(dims, (8,8), colspan=8, rowspan=3)
    plot_odd(target.lc, period.value, t0.value, depth, ax)
    plt.subplots_adjust(wspace=0)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    ax = plt.subplot2grid(dims, (12,0), colspan=8, rowspan=4)
    plot_tr_top(target.lc, model_lc, kt_period, kt_t0, ax)

    ax = plt.subplot2grid(dims, (16,0), colspan=8, rowspan=2)
    plot_tr_bottom(target.lc, model_lc, kt_period, kt_t0, ax)
    plt.subplots_adjust(hspace=0)

    ax = plt.subplot2grid(dims, (12,17), colspan=7, rowspan=3)
    plot_bls(target.lc, ax, results=bls_results)
    plt.subplots_adjust(hspace=0)

    ax = plt.subplot2grid(dims, (15,17), colspan=7, rowspan=3)
    freq, fts = plot_fft(target.lc, ax)
    plt.subplots_adjust(hspace=0)

    ax = plt.subplot2grid(dims, (13,9), colspan=6, rowspan=4)
    if target.has_target_info:
        plot_table(target, model_lc, ktransit_model, depth_snr,
                   dur, scaled_residuals, ax)
    else:
        ax.axis('off')

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches([d-1 for d in dims[::-1]])

    if save_data:
        np.savetxt(outdir+'/timeseries/'+str(target.ticid)+'.dat.ts', np.transpose([target.lc.time.value, target.lc.flux.value]), fmt='%.8f', delimiter=' ')
        np.savetxt(outdir+'/fft/'+str(target.ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')
        with open(os.path.join(outdir, "transit_stats.txt"), "a+") as file:
            file.write(f"{target.ticid} {depth} {depth_snr} {period} {t0} {dur} {scaled_residuals}\n")

    if save_fig:
        fig.savefig(str(outdir)+'/plots/'+str(target.ticid)+'_summary.png', bbox_inches='tight')

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


    target.lc.scatter(ax=ax, c='k', s=50)
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
    flux_lc.fold(per, t0).remove_outliers().bin(.1).scatter(ax=ax, c='dodgerblue', s=420)
    model_lc.fold(per, t0).plot(ax=ax, c='r', lw=3, zorder=10000)

    ax.set_ylabel('Normalized Flux')

def plot_tr_bottom(flux_lc, model_lc, per, t0, ax):
    res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
    res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)
    res_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=50)
    res_lc.fold(per, t0).remove_outliers().bin(.1).scatter(ax=ax, c='dodgerblue', s=420)
    ax.axhline(0, c='k', linestyle='dashed')
    ax.set_xlim(-.1*per, .1*per)
    ax.set_ylabel('Residuals (ppm)')

def plot_fft(lc, ax=None):
    if ax is None:
        _, ax = plt.subplots(1)

    # osample=5.
    nyq=283.
    #
    # time = lc.time
    # flux = lc.flux
    #
    # # calculate FFT
    # freq, amp, nout, jmax, prob = lomb.fasper(time, flux, osample, 3.)
    # freq = 1000. * freq / 86.4
    # bin = freq[1] - freq[0]
    # fts = 2. * amp * np.var(flux * 1e6) / (np.sum(amp) * bin)

    ls = lc.to_periodogram('ls')
    freq = ls.frequency.to(u.uHz).value
    fts = ls.power.value

    use = np.where(freq < nyq + 150)
    freq = freq[use]
    fts = fts[use]

    ax.loglog(freq, fts/np.max(fts), c='dodgerblue')
    ax.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5), color='gold', lw=2.5)
    ax.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50), color='r', lw=2.5)
    ax.axvline(283,-1,1, ls='--', color='k')
    ax.set_xlabel("Frequency [uHz]")
    ax.set_ylabel("Power")
    ax.set_xlim(10, 400)
    ax.set_ylim(1e-4, 1e0)

    return freq, fts

def get_bls_results(lc):
    model = BoxLeastSquares(lc.time, lc.flux)
    results = model.power(np.linspace(1., 25., 1000), 0.16)
    # results = model.autopower(0.16, minimum_period=1., maximum_period=25.)
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
    ax.axvline(period.value, alpha=0.4, lw=4, c='cornflowerblue')
    for n in range(2, 10):
        ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed", c='cornflowerblue')
        ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed", c='cornflowerblue')

def plot_folded(lc, period, t0, depth, ax):

    if ax is None:
        _, ax = plt.subplots(1)

    # time, flux, flux_err = lc.time, lc.flux, lc.flux_err
    #
    # phase = (t0 % period) / period
    # foldedtimes = (((time.value - phase * period) / period) % 1)
    # foldedtimes[foldedtimes > 0.5] -= 1
    # foldtimesort = np.argsort(foldedtimes)
    # foldfluxes = flux[foldtimesort]
    #
    # ax.plot(foldedtimes, flux, 'k.', markersize=2)
    # ax.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes, 40), lw=2, color='r')#, label=f'P={period:.2f} days, dur={dur:.2f} hrs')
    # ax.set_xlabel('Phase')
    # ax.set_ylabel('Flux')
    # ax.set_xlim(-0.5, 0.5)
    lc.fold(period, t0).scatter(ax=ax, c='gray', s=25,
                                label=rf'$P={period:.2f}$ d')
    lc.fold(period, t0).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(-.5*period, .5*period)
    ax.set_ylim(-3*depth, 2*depth)
    plt.grid(True)

def plot_odd(lc, period, t0, depth, ax):

    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='gray', label='Odd Transit', s=25)
    lc.fold(2*period, t0+period/2).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(0, period)
    ax.set_ylim(-3*depth, 2*depth)

    plt.grid(True)

def plot_even(lc, period, t0, depth, ax):

    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='gray', label='Even Transit', s=25)
    lc.fold(2*period, t0+period/2).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(-period, 0)
    ax.set_ylim(-3*depth, 2*depth)

    plt.grid(True)

def plot_tpf(target, ax):
    fnumber = 100
    ax = target.tpf.plot(ax=ax, show_colorbar=True, frame=fnumber, title=f'TIC {target.ticid}, cadence {fnumber}')
    ax = add_gaia_figure_elements(target.tpf, ax)

def plot_table(target, model_lc, ktransit_model, depth_snr, dur, resid, ax):
    result = ktransit_model.fitresult[1:]

    col_labels = ['Period (days)', 'b', 't0', 'Rp/Rs', r'R$_P$ (R$_J$)', 'Duration (hours)', 'Depth SNR', 'Scaled Likelihood']
    values = [f'{val:.3f}' for val in result]

    rstar = float(target.target_row['rad'].values[0])
    values.append(f'{float(values[-1]) * rstar * 9.731:.3f}')
    values.append(f'{dur.value:.3f}')
    values.append(f'{depth_snr:.3f}')
    values.append(f'{resid:.3f}')

    ax.axis('tight')
    ax.axis('off')
    tab = ax.table(list(zip(col_labels, values)), colLabels=None, loc='center', edges='open', fontsize=16)
    for r in range(0, len(col_labels)):
        cell = tab[r, 0]
        cell.set_height(0.175)
        cell = tab[r, 1]
        cell.set_height(0.175)

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

    if target.has_target_info:
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
    else:
        param_string = ''

    return param_string
