import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from astropy.stats import BoxLeastSquares
from astropy.coordinates import SkyCoord, Angle
import lightkurve as lk
import astropy.units as u
import pickle
from astroquery.mast import Catalogs

try:
    from .utils import build_ktransit_model, _individual_ktransit_dur
except:
    from utils import build_ktransit_model, _individual_ktransit_dur

__all__ = ['plot_summary']

def add_gaia_figure_elements(tpf, fig, magnitude_limit=18):
    """
    Add Gaia DR2 sources to a TPF plot.

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile
        Target pixel file to plot.
    fig : matplotlib.pyplot.figure
        Figure to plot on.
    magnitude_limit : float
        Magnitude limit to use for Gaia query.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure with Gaia sources plotted.
    """
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
    Produce a summary plot for a given target.

    Parameters
    ----------
    target : giants.Target
        Target object to plot.
    outdir : str
        Path to the output directory.
    save_data : bool
        Flag to indicate whether to save the data.
    save_fig : bool
        Flag to indicate whether to save the figure.
    """

    font = {'family' : 'sans',
        'size'   : 14}
    matplotlib.rc('font', **font)
    plt.style.use('seaborn-muted')

    dims=(18, 24)

    # fit BLS
    bls_results, bls_stats = get_bls_results(target.lc, target.ticid)
    period = bls_results.period[np.argmax(bls_results.power)]
    t0 = bls_results.transit_time[np.argmax(bls_results.power)]
    depth = bls_results.depth[np.argmax(bls_results.power)]
    depth_snr = depth / np.std(target.lc.flux.value)

    # generate ktransit fit
    model_lc, ktransit_model = fit_transit_model(target, period, t0)
    result = ktransit_model.fitresult[1:]
    kt_period = result[0]
    kt_t0 = result[2]
    dur = _individual_ktransit_dur(model_lc.time, model_lc.flux)

    scaled_residuals = np.median(fit_transit_model(target, period, t0)[1].residuals()) / np.std(target.lc.flux.value)

    """Create the figure."""
    fig = plt.gcf()
    fig.suptitle(f'TIC {target.ticid}', fontweight='bold', size=24, y=0.93)

    # plot the light curve
    ax = plt.subplot2grid(dims, (0,0), colspan=24, rowspan=3)
    plot_raw_lc(target, model_lc, ax)

    # set title to include stellar params
    param_string = stellar_params(target)
    ax.set_title(param_string, size=20)

    # plot the folded light curve
    ax = plt.subplot2grid(dims, (4,0), colspan=16, rowspan=3)
    plot_folded(target.lc, period.value, t0.value, depth, ax)

    # plot the TPF
    ax = plt.subplot2grid(dims, (4,17), colspan=7, rowspan=7)
    plot_tpf(target, ax)

    # plot the odd and even transits
    ax = plt.subplot2grid(dims, (8,0), colspan=8, rowspan=3)
    plot_even(target.lc, period.value, t0.value, depth, ax)
    ax = plt.subplot2grid(dims, (8,8), colspan=8, rowspan=3)
    plot_odd(target.lc, period.value, t0.value, depth, ax)
    plt.subplots_adjust(wspace=0)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # plot the transit model
    ax = plt.subplot2grid(dims, (12,0), colspan=8, rowspan=4)
    plot_tr_top(target.lc, model_lc, kt_period, kt_t0, ax)

    # plot the residuals
    ax = plt.subplot2grid(dims, (16,0), colspan=8, rowspan=2)
    plot_tr_bottom(target.lc, model_lc, kt_period, kt_t0, ax)
    plt.subplots_adjust(hspace=0)

    # plot the BLS periodogram
    ax = plt.subplot2grid(dims, (12,17), colspan=7, rowspan=3)
    plot_bls(target.lc, ax, results=bls_results)
    plt.subplots_adjust(hspace=0)

    # plot the FFT
    ax = plt.subplot2grid(dims, (15,17), colspan=7, rowspan=3)
    freq, fts = plot_fft(target.lc, ax)
    plt.subplots_adjust(hspace=0)

    # include the transit stats table
    ax = plt.subplot2grid(dims, (13,9), colspan=6, rowspan=4)
    if target.has_target_info:
        plot_table(target, ktransit_model, depth_snr,
                   dur, scaled_residuals, ax)
    else:
        ax.axis('off')

    harmonic_del = bls_stats['harmonic_delta_log_likelihood'].value
    sde = (bls_results.power - np.mean(bls_results.power)) / np.std(bls_results.power)
    max_power = max(sde)

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches([d-1 for d in dims[::-1]])

    # save the transit stats
    with open(os.path.join(outdir, "transit_stats.txt"), "a+") as file:
                file.write(f"{target.ticid} {depth} {depth_snr} {period} {t0} {dur} {scaled_residuals} {harmonic_del} {max_power}\n")

    # save the data
    if save_data:
        try:
            np.savetxt(outdir+'/timeseries/'+str(target.ticid)+'.dat.ts', np.transpose([target.lc.time.value, target.lc.flux.value]), fmt='%.8f', delimiter=' ')
            np.savetxt(outdir+'/fft/'+str(target.ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')
        except:
            np.savetxt(outdir+str(target.ticid)+'.dat.ts', np.transpose([target.lc.time.value, target.lc.flux.value]), fmt='%.8f', delimiter=' ')
            np.savetxt(outdir+str(target.ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')

    if save_fig:
        try:
            fig.savefig(str(outdir)+'/plots/'+str(target.ticid)+'_summary.png', bbox_inches='tight')
        except:
            fig.savefig(str(outdir)+str(target.ticid)+'_summary.png', bbox_inches='tight')

def fit_transit_model(target, period, t0):
    """
    Fit a transit model to a given target using the ktransit package.

    Parameters
    ----------
    target : giants.Target
        Target object to fit.
    period : float
        Orbital period of the planet.
    t0 : float
        Epoch of the first transit.

    Returns
    -------
    model_lc : lightkurve.LightCurve
        Light curve of the transit model.
    ktransit_model : ktransit.ktransit.LCModel
        ktransit model object.
    """

    ktransit_model = build_ktransit_model(target.ticid, target.lc, period, t0)

    model_lc = lk.LightCurve(time=target.lc.time, flux=ktransit_model.transitmodel)
    return model_lc, ktransit_model

def plot_raw_lc(target, model_lc, ax=None):
    """
    Plot the raw light curve of a given target.

    Parameters
    ----------
    target : giants.Target
        Target object to plot.
    model_lc : lightkurve.LightCurve
        Light curve of the transit model.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    if ax is None:
        _, ax = plt.subplots(1)

    target.lc.scatter(ax=ax, c='k', s=50)
    ax.set_xlim(target.lc.time.value[0], target.lc.time.value[-1])
    for b in target.breakpoints:
        try:
            ax.axvline(b.value, linestyle='--', color='r')
        except:
            ax.axvline(b, linestyle='--', color='r')

    depth = 0 - np.min(model_lc.flux.value)
    ax.set_ylim(np.min(model_lc.flux.value)-depth*2, depth*2)

def plot_tr_top(flux_lc, model_lc, per, t0, ax):
    """
    Plot the transit model on top of the raw light curve.    

    Parameters
    ----------
    flux_lc : lightkurve.LightCurve
        Light curve of the target.
    model_lc : lightkurve.LightCurve
        Light curve of the transit model.
    per : float
        Orbital period of the planet.
    t0 : float
        Epoch of the first transit.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
    res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)

    depth = 0 - np.min(model_lc.flux.value)

    ax.set_xticklabels([])
    ax.set_xlim(-.1*per, .1*per)

    flux_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=50)
    flux_lc.fold(per, t0).remove_outliers().bin(.1).scatter(ax=ax, c='dodgerblue', s=420)
    model_lc.fold(per, t0).plot(ax=ax, c='r', lw=3, zorder=10000)

    ax.set_ylim(np.min(model_lc.flux.value)-depth*2, depth*2)
    ax.set_ylabel('Normalized Flux')

def plot_tr_bottom(flux_lc, model_lc, per, t0, ax):
    """
    Plot the residuals of the transit model.
    
    Parameters
    ----------
    flux_lc : lightkurve.LightCurve
        Light curve of the target.
    model_lc : lightkurve.LightCurve
        Light curve of the transit model.
    per : float
        Orbital period of the planet.
    t0 : float
        Epoch of the first transit.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
    res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)
    res_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=50)
    res_lc.fold(per, t0).remove_outliers().bin(.1).scatter(ax=ax, c='dodgerblue', s=420)
    ax.axhline(0, c='k', linestyle='dashed')
    ax.set_xlim(-.1*per, .1*per)
    ax.set_ylabel('Residuals (ppm)')

def plot_fft(lc, ax=None):
    """
    Plot the FFT of a given light curve.
    
    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to plot.
    ax : matplotlib.pyplot.axis
        Axis to plot on.

    Returns
    -------
    freq : numpy.ndarray
        Frequencies of the FFT.
    fts : numpy.ndarray
        Power of the FFT.
    """
    if ax is None:
        _, ax = plt.subplots(1)

    nyq=283.
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

def get_bls_results(lc, targetid='None'):
    """
    Get the BLS results for a given light curve.
    
    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to fit.
        
    Returns
    -------
    results : astropy.stats.BoxLeastSquares
        Astropy BLS object.        
    """
    model = BoxLeastSquares(lc.time, lc.flux)
    results = model.power(np.linspace(1., 25., 1000), 0.16)

    stats = model.compute_stats(results.period[np.argmax(results.power)], 
                                results.duration[np.argmax(results.power)], 
                                results.transit_time[np.argmax(results.power)])
    
    stats['period'] = results.period[np.argmax(results.power)]
    stats['duration'] = results.duration[np.argmax(results.power)]

    return results, stats

def plot_bls(lc, ax, results=None):
    """
    Plot the BLS periodogram for a given light curve.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to plot.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    results : astropy.stats.BoxLeastSquares
        Astropy BLS object.
    """
    if results is None:
        results, stats = get_bls_results(lc)
    period = results.period[np.argmax(results.power)]

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
    """
    Plot the folded light curve for a given light curve.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to plot.
    period : float
        Orbital period of the planet.
    t0 : float
        Epoch of the first transit.
    depth : float
        Depth of the transit.
    ax : matplotlib.pyplot.axis
        Axis to plot on.  
    """
    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(period, t0).scatter(ax=ax, c='gray', s=25,
                                label=rf'$P={period:.2f}$ d')
    lc.fold(period, t0).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(-.5*period, .5*period)
    ax.set_ylim(-3*depth, 2*depth)
    plt.grid(True)

def plot_odd(lc, period, t0, depth, ax):
    """
    Plot the odd transits for a given light curve.
    
    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to plot.
    period : float
        Orbital period of the planet.
    t0 : float
        Epoch of the first transit.
    depth : float
        Depth of the transit.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='gray', label='Odd Transit', s=25)
    lc.fold(2*period, t0+period/2).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(0, period)
    ax.set_ylim(-3*depth, 2*depth)

    plt.grid(True)

def plot_even(lc, period, t0, depth, ax):
    """
    Plot the even transits for a given light curve.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to plot.
    period : float
        Orbital period of the planet.
    t0 : float
        Epoch of the first transit.
    depth : float
        Depth of the transit.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    if ax is None:
        _, ax = plt.subplots(1)

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='gray', label='Even Transit', s=25)
    lc.fold(2*period, t0+period/2).bin(.1).plot(ax=ax, c='r', lw=2)
    ax.set_xlim(-period, 0)
    ax.set_ylim(-3*depth, 2*depth)

    plt.grid(True)

def plot_tpf(target, ax):
    """
    Plot the TPF for a given target.

    Parameters
    ----------
    target : giants.Target
        Target object to plot.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    fnumber = 100
    ax = target.tpf.plot(ax=ax, show_colorbar=True, frame=fnumber, title=f'TIC {target.ticid}, cadence {fnumber}')
    ax = add_gaia_figure_elements(target.tpf, ax)

def plot_table(target, ktransit_model, depth_snr, dur, resid, ax):
    """
    Include a table of transit parameters in the summary plot.

    Parameters
    ----------
    target : giants.Target
        Target object to plot.
    ktransit_model : ktransit.ktransit.LCModel
        ktransit model object.
    depth_snr : float
        Depth SNR of the transit.
    dur : astropy.units.quantity.Quantity
        Duration of the transit.
    resid : float
        Scaled residuals of the transit.
    ax : matplotlib.pyplot.axis
        Axis to plot on.
    """
    result = ktransit_model.fitresult[1:]

    col_labels = ['Period (days)', 'b', 't0', 'Rp/Rs', r'R$_P$ (R$_J$)', 'Duration (hours)', 'Depth SNR', 'Scaled Likelihood']
    values = [f'{val:.3f}' for val in result]

    values.append(f'{float(values[-1]) * target.rstar * 9.731:.3f}')
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
    """
    Retrieve stellar parameters for a given target.

    Parameters
    ----------
    target : giants.Target
        Target object to plot.

    Returns
    -------
    param_string : str
        String of stellar parameters.
    """
    catalog_data = Catalogs.query_criteria(objectname=f'TIC {target.ticid}', catalog="Tic", radius=.0001, Bmag=[0,20])
    
    ra = catalog_data['ra'][0]
    dec = catalog_data['dec'][0]
    coords = f'({ra:.2f}, {dec:.2f})'
    rstar = catalog_data['rad'][0]
    teff = catalog_data['Teff'][0]
    if np.isnan(rstar):
        rstar = '?'
    else:
        rstar = f'{rstar:.2f}'
    if np.isnan(teff):
        teff = '?'
    else:
        teff = f'{teff:.0f}'
    logg = catalog_data['logg'][0]
    if np.isnan(logg):
        logg = '?'
    else:
        logg = f'{logg:.2f}'
    V = catalog_data['Vmag'][0]

    param_string = rf'(RA, dec)={coords}, R_star={rstar} $R_\odot$, logg={logg}, Teff={teff} K, V={float(V):.2f}'

    return param_string
