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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    from .utils import build_ktransit_model, parse_sectors
except:
    from utils import build_ktransit_model, parse_sectors

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

def plot_summary(target, outdir='', save_data=False, save_fig=True, 
                 custom_lc=None, custom_id=None):
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

    if custom_lc is not None:
        lc = custom_lc
        lcc, sectors = parse_sectors(lc)
        ticid = custom_id
        tpf = lk.search_tesscut('TIC '+str(ticid))[0].download(cutout_size=11)
        aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)

    else:
        lc = target.lc
        lcc = target.lcc
        ticid = target.ticid
        tpf = target.tpf
        aperture_mask = target.aperture_mask
        sectors = target.used_sectors

        mask = target.link_mask[~target.mask]
        try:
            target.lc = target.lc[mask] # PHT HACK
        except:
            target.lc = target.lc

    freq, fts = calculate_fft(lc)

    # save the data
    if save_data:
        try:
            np.savetxt(outdir+'/timeseries/'+str(ticid)+'.dat.ts', np.transpose([lc.time.value, lc.flux.value]), fmt='%.8f', delimiter=' ')
            np.savetxt(outdir+'/fft/'+str(ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')
        except:
            np.savetxt(outdir+str(ticid)+'.dat.ts', np.transpose([lc.time.value, lc.flux.value]), fmt='%.8f', delimiter=' ')
            np.savetxt(outdir+str(ticid)+'.dat.ts.fft', np.transpose([freq, fts]), fmt='%.8f', delimiter=' ')

    # fit BLS
    bls_results, bls_stats = get_bls_results(lc)
    period = bls_results.period[np.argmax(bls_results.power)]
    t0 = bls_results.transit_time[np.argmax(bls_results.power)]
    depth = bls_results.depth[np.argmax(bls_results.power)]
    depth_snr = depth / np.std(lc.flux.value)
    dur = bls_stats['duration'].value * 24.

    harmonic_del = bls_stats['harmonic_delta_log_likelihood'].value
    sde = (bls_results.power - np.mean(bls_results.power)) / np.std(bls_results.power)
    max_power = max(sde)

    try:
        # generate ktransit fit
        model_lc, ktransit_model = fit_transit_model(lc, period, t0)
        result = ktransit_model.fitresult[1:]
        kt_period = result[0]
        kt_t0 = result[2]

        scaled_residuals = np.median(ktransit_model.residuals()) / np.std(lc.flux.value)

    except:
        model_lc = None
        kt_period = period
        kt_t0 = t0

        scaled_residuals = np.nan

    # save the transit stats
    with open(os.path.join(outdir, "transit_stats.txt"), "a+") as file:
                file.write(f"{ticid} {depth} {depth_snr} {period} {t0} {dur} {scaled_residuals} {harmonic_del} {max_power}\n")

    """Create the figure."""
    dims = (27, 36)
    fig = plt.figure(figsize=dims[::-1], dpi=250)
    ax_top = plt.subplot2grid(dims, (0, 0), colspan=dims[1], rowspan=1)
    ax_top.axis('off')
    ax_bot = plt.subplot2grid(dims, (dims[0]-1, 0), colspan=dims[1], rowspan=1)
    ax_bot.axis('off')
    ax_top.set_title(f'TIC {ticid}', fontweight='bold', size=24, y=0.93)
    
    # set title to include stellar params
    param_string, rstar = stellar_params(ticid)
    ax_top.annotate(param_string, size=20, xy=(0.5, 0.65), xycoords='axes fraction', ha='center', va='center')

    # plot the raw light curve
    ax = plt.subplot2grid(dims, (1, 0), colspan=dims[1], rowspan=4)
    ax.axis('off')
    plot_raw_lc(lcc, sectors, model_lc, kt_period, kt_t0, depth, ax)

    # plot the folded light curve
    ax = plt.subplot2grid(dims, (7,0), colspan=18, rowspan=4)
    plot_folded(lc, kt_period, kt_t0, depth, dur, ax)

    # plot the odd and even transits
    ax = plt.subplot2grid(dims, (12,0), colspan=9, rowspan=4)
    plot_even(lc, kt_period, kt_t0, depth, dur, ax)
    ax = plt.subplot2grid(dims, (12,9), colspan=9, rowspan=4)
    plot_odd(lc, kt_period, kt_t0, depth, dur, ax)
    plt.subplots_adjust(wspace=0)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # include the transit stats table
    ax = plt.subplot2grid(dims, (7,19), colspan=5, rowspan=9)
    plot_table(ktransit_model, depth_snr, dur, scaled_residuals, rstar, ax)

    # plot the TPF
    ax = plt.subplot2grid(dims, (7,25), colspan=11, rowspan=9)
    plot_tpf(tpf, ticid, aperture_mask, ax)

    # plot the transit model
    ax = plt.subplot2grid(dims, (18,0), colspan=12, rowspan=6)
    plot_tr_top(lc, model_lc, kt_period, kt_t0, depth, ax)

    # plot the residuals
    ax = plt.subplot2grid(dims, (24,0), colspan=12, rowspan=3)
    plot_tr_bottom(lc, model_lc, kt_period, kt_t0, depth, ax)
    plt.subplots_adjust(hspace=0)

    # plot the BLS periodogram
    ax = plt.subplot2grid(dims, (18,14), colspan=10, rowspan=4)
    plot_bls(lc, ax, results=bls_results)
    plt.subplots_adjust(hspace=0)

    # plot the FFT
    ax = plt.subplot2grid(dims, (23,14), colspan=10, rowspan=4)
    plot_fft(freq, fts, ax)
    plt.subplots_adjust(hspace=0)

    # plot the difference image
    ax = plt.subplot2grid(dims, (18, 25), colspan=11, rowspan=9)
    plot_diff_image(tpf, lcc, kt_period, kt_t0, dur, ax)

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    # fig.set_size_inches([d-1 for d in dims[::-1]])
    fig.set_size_inches([33, 25.5])

    if save_fig:
        try:
            fig.savefig(str(outdir)+'/plots/'+str(ticid)+'_summary.png', bbox_inches='tight')
        except:
            fig.savefig(str(outdir)+str(ticid)+'_summary.png', bbox_inches='tight')

def fit_transit_model(lc, period, t0):
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

    ktransit_model = build_ktransit_model(lc, period, t0)

    model_lc = lk.LightCurve(time=lc.time, flux=ktransit_model.transitmodel)
    return model_lc, ktransit_model

def plot_raw_lc(lcc, sectors, model_lc, per, t0, depth, ax=None):
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

    for n in range(len(lcc)):
        i = len(lcc) - n - 1
        lc = lcc[i]
        ax_in = inset_axes(ax, width=f'{100/len(lcc)}%', height='100%', 
                           bbox_to_anchor=(( - n) / len(lcc), 0, 1, 1), 
                           bbox_transform=ax.transAxes)
        lc.scatter(ax=ax_in, c='gray', s=75, alpha=0.4, edgecolor='None')

        n_transits = round((lcc[-1].time.value[-1] - lcc[0].time.value[0]) / per)
        transit_times = t0 + np.arange(n_transits) * per
        tt_masked = transit_times[(transit_times > lc.time.value[0]) & (transit_times < lc.time.value[-1])]
        tt_y = np.ones_like(tt_masked) * .9 * (np.min(model_lc.flux.value)-depth*4)

        ax_in.scatter(tt_masked, tt_y, color='r', marker='^', s=75, edgecolors='None', zorder=10000)
        if len(sectors) < 15:
            ax_in.set_title(f'Sector {sectors[i]}')
            ax_in.set_xlabel('Time [BTJD]')
        else:
            ax_in.set_title(f'{sectors[i]}')
            ax_in.set_xlabel('Time')
        ax_in.set_ylim(np.min(model_lc.flux.value)-depth*4, depth*4)
        ax_in.set_xlim(lc.time.value[0]-.5, lc.time.value[-1]+.5)
        if i > 0:
            ax_in.set_ylabel('')
            ax_in.set_yticklabels([])
            ax_in.spines['left'].set_visible(False)
        if i < len(lcc) - 1:
            ax_in.spines['right'].set_linestyle((0,(8,5)))

def plot_tr_top(flux_lc, model_lc, per, t0, depth, ax):
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

    flux_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=125, alpha=0.4, edgecolor='None')
    flux_lc.fold(per, t0).remove_outliers().bin(.125).scatter(ax=ax, c='dodgerblue', s=720, edgecolor='k')
    
    if model_lc is not None:

        res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
        res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)

        model_lc.fold(per, t0).plot(ax=ax, c='r', lw=3, zorder=10000)
        ax.set_ylim(np.min(model_lc.flux.value)-depth*2, depth*2)

    ax.set_xticklabels([])
    ax.set_xlim(-.15*per, .15*per)
    
    ax.set_ylabel('Normalized Flux')

def plot_tr_bottom(flux_lc, model_lc, per, t0, depth, ax):
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
    if model_lc is not None:
  
        res_flux_ppm = (flux_lc.flux - model_lc.flux.reshape(len(flux_lc.flux))) * 1e6
        res_lc = lk.LightCurve(time=model_lc.time, flux=res_flux_ppm)
        res_lc.fold(per, t0).remove_outliers().scatter(ax=ax, c='gray', s=125, alpha=0.4, edgecolor='None')
        res_lc.fold(per, t0).remove_outliers().bin(.125).scatter(ax=ax, c='dodgerblue', s=720, edgecolor='k')
        ax.set_ylim((-depth*2)*1e6, (depth*2)*1e6)

    else:
        ax.annotate('No transit model', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=20, ha='center', va='center')

    ax.axhline(0, c='k', linestyle='dashed')
    ax.set_xlim(-.15*per, .15*per)
    ax.set_ylabel('Residuals (ppm)')

def calculate_fft(lc):
    nyq=283.
    ls = lc.to_periodogram('ls')
    freq = ls.frequency.to(u.uHz).value
    fts = ls.power.value

    use = np.where(freq < nyq + 150)
    freq = freq[use]
    fts = fts[use]

    return freq, fts

def plot_fft(freq, fts, ax=None):
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

    ax.loglog(freq, fts/np.max(fts), c='dodgerblue')
    ax.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 5), color='gold', lw=1.5)
    ax.loglog(freq, scipy.ndimage.filters.gaussian_filter(fts/np.max(fts), 50), color='r', lw=1.5)
    ax.axvline(283,-1,1, ls='--', color='k')
    ax.set_xlabel("Frequency [uHz]")
    ax.set_ylabel("Power")
    ax.set_xlim(10, 400)
    ax.set_ylim(1e-4, 1e0)

    return freq, fts

def get_bls_results(lc):
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

    try:
        lc = lc.bin(.5/24).remove_nans()
    except:
        lc = lc.remove_nans()
        
    # create boolean mask for tpf
    link_mask = np.ones_like(lc.time.value, dtype=bool)

    # add the first 24 and last 12 hours of data to mask
    link_mask[lc.time.value < lc.time.value[0] + 1.0] = False
    link_mask[lc.time.value > lc.time.value[-1] - 0.5] = False

    # identify the largest gap in the data
    gap = np.argmax(np.diff(lc.time.value))

    # mask 24 hours after and 12 hours before the largest gap
    link_mask[(lc.time.value < lc.time.value[gap] + 1.0) & (lc.time.value > lc.time.value[gap] - 0.5)] = False

    # drop False indicies from lc
    lc = lc[link_mask]

    model = BoxLeastSquares(lc.time, lc.flux)
    results = model.power(np.linspace(1., 25., 7000), np.linspace(.1, .5, 1000))

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

    ax.plot(results.period, results.power, "k", lw=0.75)
    ax.set_xlim(results.period.min().value, results.period.max().value)
    ax.set_xlabel("period [days]")
    ax.set_ylabel("log likelihood")

    # Highlight the harmonics of the peak period
    ax.axvline(period.value, alpha=0.4, lw=4, c='cornflowerblue')
    for n in range(2, 10):
        ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed", c='cornflowerblue')
        ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed", c='cornflowerblue')

def plot_folded(lc, period, t0, depth, dur, ax):
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

    lc.fold(period, t0).scatter(ax=ax, c='gray', s=75, alpha=0.4,
                                label=rf'$P={period:.2f}$ d', edgecolor='None')
    lc.fold(period, t0).bin(.075).plot(ax=ax, c='r', lw=2, zorder=1001)
    ax.scatter(0, -.9 * 3 * depth, c='r', s=150, edgecolors='k', marker='^', zorder=1002)
    ax.plot([-(dur/24.)/2, (dur/24.)/2], [-.9 * 3 * depth, -.9 * 3 * depth], 'r', lw=1.5, zorder=1000)

    ax.set_xlim(-.5*period, .5*period)
    ax.set_ylim(-3*depth, 2*depth)
    ax.legend(loc='upper right', fontsize=18)
    plt.grid(True)

def plot_odd(lc, period, t0, depth, dur, ax):
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

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='gray', label='Odd Transit', s=75, alpha=0.4, edgecolor='None')
    lc.fold(2*period, t0+period/2).bin(.075).plot(ax=ax, c='r', lw=2, zorder=1002)
    ax.scatter(period/2, -.9 * 3 * depth, c='r', s=150, edgecolors='k', marker='^', zorder=1001)
    ax.plot([.5*period-(dur/24.)/2, .5*period+(dur/24.)/2], [-.9 * 3 * depth, -.9 * 3 * depth], 'r', lw=1.5, zorder=1000)

    ax.set_xlim(0, period)
    ax.set_ylim(-3*depth, 2*depth)
    ax.legend(loc='upper right', fontsize=18)

    plt.grid(True)

def plot_even(lc, period, t0, depth, dur, ax):
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

    lc.fold(2*period, t0+period/2).scatter(ax=ax, c='gray', label='Even Transit', s=75, alpha=0.4, edgecolor='None')
    lc.fold(2*period, t0+period/2).bin(.075).plot(ax=ax, c='r', lw=2, zorder=1002)
    ax.scatter(-period/2, -.9 * 3 * depth, c='r', s=150, edgecolors='k', marker='^', zorder=1001)
    ax.plot([-.5*period-(dur/24.)/2, -.5*period+(dur/24.)/2], [-.9 * 3 * depth, -.9 * 3 * depth], 'r', lw=1.5, zorder=1000)

    ax.set_xlim(-period, 0)
    ax.set_ylim(-3*depth, 2*depth)
    ax.legend(loc='upper right', fontsize=18)

    plt.grid(True)

def plot_tpf(tpf, ticid, aperture_mask, ax):
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
    ax = tpf.plot(ax=ax, show_colorbar=True, frame=fnumber, 
                         aperture_mask=aperture_mask, mask_color='k',
                         title=f'TIC {ticid}, cadence {fnumber}')
    ax = add_gaia_figure_elements(tpf, ax)

def plot_diff_image(tpf, lcc, per, t0, dur, ax):
    """
    Plot the difference image for a given target.
    """

    n_transits = round((lcc[-1].time.value[-1] - lcc[0].time.value[0]) / per)
    transit_times = t0 + np.arange(n_transits) * per

    resid_frames = []
    try:
        for tt in transit_times[transit_times < tpf.time.value[-1]]:
            try:
                t_frames = np.where([np.min(np.abs(tt - t)) < (dur/2)/24. for t in tpf.time.value])
                nt_frames = np.where([np.min(np.abs(tt + (dur/24.) - t)) < (dur/2)/24. for t in tpf.time.value])
                rf = np.nanmean(tpf.flux.value[t_frames], axis=0) - np.nanmean(tpf.flux.value[nt_frames], axis=0)
                if np.nansum(rf) == 0:
                    continue
                else:
                    resid_frames.append(rf)
            except:
                continue

        residual = np.nanmedian(resid_frames, axis=0)
        mappable = plt.imshow(residual)
        plt.xlim(-.5, 10.5)
        plt.ylim(-.5, 10.5)
        plt.colorbar(mappable, label='Residual Counts')
        ax.set_title('(In - Out) Transit')
    except:
        tpf.plot(frame=100, ax=ax)
        ax.annotate('Difference Image Failed', xy=(0.5, 0.1), xycoords='axes fraction', 
                    fontsize=28, ha='center', va='center', backgroundcolor='w', 
                    bbox=dict(facecolor='w', alpha=0.75, edgecolor='black'))

    plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_in_out_check(target, period, t0, depth, ax):
    """
    Plot the in-transit and out-of-transit check for a given target.
    """
    in_index = np.argmin(np.absolute(target.lc.time.value-t0))
    out_index = np.argmin(np.absolute(target.lc.time.value-(t0+period/4.)))

    target.lc.plot(ax=ax, c='k', lw=1, label=None)
    ax.scatter(target.lc.time.value[in_index], -.9 * 3 * depth, 
               c='r', s=150, edgecolors='k', marker='^', zorder=1000, label='In')
    ax.scatter(target.lc.time.value[out_index], -.9 * 3 * depth, 
               c='cornflowerblue', s=150, edgecolors='k', marker='^', zorder=1000, label='Out')
    ax.set_xlim(target.lc.time.value[in_index]-.5*period, target.lc.time.value[out_index]+.5*period)
    ax.set_ylim(-3*depth, 2*depth)

    ax.legend(loc='lower right', fontsize=16)

def plot_table(ktransit_model, depth_snr, dur, resid, rstar, ax):
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

    col_labels = ['Period (days)', r'$b$', r'$t_0$ (BTJD)', r'$R_p$ / $R_\bigstar$', r'$R_P$ ($R_J$)', 'Duration (hours)', 'Depth SNR', 'Scaled Likelihood']
    values = [f'{np.abs(val):.5f}' for val in result]

    values.append(f'{float(values[-1]) * rstar * 9.731:.3f}')
    values.append(f'{dur:.3f}')
    values.append(f'{depth_snr:.3f}')
    values.append(f'{resid:.3f}')

    ax.axis('tight')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.axis('off')
    tab = ax.table(list(zip(col_labels, values)), colLabels=None, loc='center')
    for r in range(0, len(col_labels)):
        cell = tab[r, 0]
        cell.set_height(0.1225)
        cell = tab[r, 1]
        cell.set_height(0.1225)
    tab.set_fontsize(90)

def stellar_params(ticid):
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
    catalog_data = Catalogs.query_criteria(objectname=f'TIC {ticid}', catalog="Tic", radius=.0001, Bmag=[0,20])
    
    ra = catalog_data['ra'][0]
    dec = catalog_data['dec'][0]
    coords = f'({ra:.2f}, {dec:.2f})'
    rstar_val = catalog_data['rad'][0]
    teff = catalog_data['Teff'][0]
    if np.isnan(rstar_val):
        rstar = '?'
    else:
        rstar = f'{rstar_val:.2f}'
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

    teffstring = r'T$_{\rm eff}$'
    param_string = rf'(RA, dec)={coords}, $R_\bigstar$={rstar} $R_\odot$, logg={logg}, {teffstring}={teff} K, V={float(V):.2f}'

    return param_string, rstar_val
