import os
import re
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares, mad_std, LombScargle
import astropy.stats as ass
import lightkurve as lk
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u
import ktransit
from matplotlib.backends.backend_pdf import PdfPages
import eleanor
try:
    from . import lomb
    from . import PACKAGEDIR
    from .plotting import plot_quicklook, plot_transit_vetting, make_ica_plot, plot_summary
    from .utils import get_cutout
except:
    import lomb
    from plotting import plot_quicklook, plot_transit_vetting, make_ica_plot, plot_summary
    from utils import get_cutout
    PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

#optional imports
try:
    from sklearn.decomposition import FastICA
except:
    pass

# suppress verbose astropy warnings and future warnings
warnings.filterwarnings("ignore", module="astropy")
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ['Giant']

class Giant(object):
    """An object to store and analyze time series data for giant stars.
    """
    def __init__(self, ticid, csv_path='data/ticgiants_bright_v2_skgrunblatt.csv', cache_path=None):
        self.PACKAGEDIR = PACKAGEDIR
        if (csv_path is None) or (csv_path==''):
            self.target_list = []
        else:
            self.target_list = self.get_targets(csv_path)
        self.lc_exists = False
        self.ticid = ticid
        self.cache_path = cache_path
        self.silent = False
        if isinstance(self.ticid, str):
            self.ticid = int(re.search(r'\d+', str(self.ticid)).group())

    def get_targets(self, csv_path='data/ticgiants_bright_v2_skgrunblatt.csv'):
        """Read in a csv of CVZ targets from a local file.
        """
        path = os.path.abspath(os.path.abspath(os.path.join(self.PACKAGEDIR, csv_path)))
        table = pd.read_csv(path, skiprows=3, dtype='unicode')

        return table

    def get_target_list(self):
        """Helper function to fetch a list of TIC IDs.

        Returns
        -------
        IDs : list
            TIC IDs for bright targets in list.
        """
        return self.target_list.ID.values

    def from_eleanor(self, save_postcard=False):
        """Download light curves from Eleanor for desired target. Eleanor light
        curves include:
        - raw : raw flux light curve
        - corr : corrected flux light curve

        Returns
        -------
        LightCurveCollection :
            ~lightkurve.LightCurveCollection containing raw and corrected light curves.
        """

        # search TESScut to figure out which sectors you need (there's probably a better way to do this)
        sectors = self._find_sectors(self.ticid)
        if not self.silent:
            print(f'Creating light curve for target {self.ticid} for sectors {sectors}.')
        # download target data for the desired source for only the first available sector

        star = eleanor.Source(tic=self.ticid, sector=int(sectors[0]), tc=True)
        try:
            data = eleanor.TargetData(star, height=11, width=11, bkg_size=27, do_psf=False, do_pca=False, try_load=True, save_postcard=save_postcard)
        except:
            data = eleanor.TargetData(star, height=7, width=7, bkg_size=21, do_psf=False, do_pca=False, try_load=True, save_postcard=save_postcard)
        q = data.quality == 0
        # create raw flux light curve
        raw_lc = lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q],label='raw', time_format='btjd').remove_nans().normalize()
        corr_lc = lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], label='corr', time_format='btjd').remove_nans().normalize()

        #track breakpoints between sectors
        self.breakpoints = [raw_lc.time[-1]]
        # iterate through extra sectors and append the light curves
        if len(sectors) > 1:
            for s in sectors[1:]:
                try: # some sectors fail randomly
                    star = eleanor.Source(tic=self.ticid, sector=int(s), tc=True)
                    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False, try_load=True)
                    q = data.quality == 0

                    raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())

                    self.breakpoints.append(raw_lc.time[-1])
                except:
                    continue
        # store in a LightCurveCollection object and return
        return lk.LightCurveCollection([raw_lc, corr_lc])

    def from_lightkurve(self, sectors=None, **kwargs):
        """
        Use `lightkurve.search_tesscut` to query and download TESSCut 11x11 cutout for target.
        This function creates a background model and subtracts it off using `lightkurve.RegressionCorrector`.

        Parameters
        ----------
        sectors : int, list of ints
            desired sector number or list of sector numbers

        Returns
        -------
        lc : `lightkurve.LightCurve` object
            background-corrected flux time series
        """
        if sectors is None:
            # search TESScut to figure out which sectors you need (there's probably a better way to do this)
            sectors = self._find_sectors(self.ticid)
        if not self.silent:
            print(f'Creating light curve for target {self.ticid} for sector(s) {sectors}.')

        search = lk.search_tesscut(f'TIC {self.ticid}', sector=sectors)
        tpfc = lk.TargetPixelFileCollection([])
        if self.cache_path is None:
            for sector in search:
                try:
                    tpfc.append(sector.download(cutout_size=11))
                except:
                    continue
        else:
            for sector in search:
                try:
                    tpfc.append(sector.download(cutout_size=11, download_dir=self.cache_path))
                except:
                    continue

        self.tpf = tpfc[0]
        lc = self.simple_pca(self.tpf)

        # store as LCC for plotting later
        self.lcc = lk.LightCurveCollection([lc])
        self.breakpoints = [lc.time[-1]]
        for tpf in tpfc[1:]:
            new_lc = self.simple_pca(tpf)
            self.breakpoints.append(new_lc.time[-1])
            self.lcc.append(new_lc)
            lc = lc.append(new_lc)

        self.lc = lc

        return lc

    def _find_sectors(self, ticid):
        """Hidden function to read sectors from a search result."""
        sectors = []

        search_result = lk.search_tesscut(f'TIC {ticid}')
        for sector in search_result.table['description']:
            sectors.append(int(re.search(r'\d+', sector).group()))
        return sectors


    def _clean_data(self, lc, gauss_filter_lc=True):
        """Hidden function to remove common sources of noise and outliers."""
        # mask first 12h after momentum dump
        momdump = (lc.time > 1339) * (lc.time < 1341)

        # also the burn in
        burnin = np.zeros_like(lc.time, dtype=bool)
        burnin[:30] = True
        downlinks = [1339.6770629882812, 1368.6353149414062, 1421.239501953125, 1451.5728759765625, 1478.114501953125,
                     1504.7199096679688, 1530.2824096679688, 1535.0115966796875, 1556.74072265625, 1582.7824096679688,
                     1610.8031616210938, 1640.0531616210938, 1668.6415405273438, 1697.3673095703125, 1724.9667358398438,
                     1751.6751098632812]
        """
        # mask around downlinks
        for d in downlinks:
            if d in lc.time:
                burnin[d:d+15] = True
        """
        # also 6 sigma outliers
        _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        mask = momdump | outliers | burnin
        lc = lc[~mask]
        lc.flux = lc.flux - 1
        if gauss_filter_lc:
            lc.flux = lc.flux - scipy.ndimage.filters.gaussian_filter(lc.flux, 100) # <2-day (5muHz) filter

        # store cleaned lc
        self.lc = lc
        self.mask = mask
        return lc


    def fetch_and_clean_data(self, lc_source='eleanor', sectors=None, gauss_filter_lc=True, **kwargs):
        """
        Query and download data, remove background signal and outliers. The light curve is stored as a
        object variable `Giant.lc`.

        Parameters
        ----------
        lc_source : str, 'lightkurve' or 'eleanor'
            pipeline used to access data
        sectors : int, list of ints
            desired sector number or list of sector numbers
        gauss_filer_lc : bool
            optionally apply Gaussian smoothing with a ~2 day filter (good for planets, bad for stars)
        """
        if lc_source == 'eleanor':
            lcc = self.from_eleanor(**kwargs)
            lc = lcc[1] # using corr_lc
            time = lc.time
            flux = lc.flux
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        elif lc_source == 'lightkurve':
            lc = self.from_lightkurve(sectors=sectors)

        lc = self._clean_data(lc)


    def vet_transit(self, lc=None, tpf=None, **kwargs):
        """
        A quick vetting method using ICA.
        """
        if not self.lc_exists:
            lcc = self.from_eleanor(**kwargs)
            lc = lcc[1] # using corr_lc
            time = lc.time
            flux = lc.flux
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
            self.lc = self._clean_data(lc)

        fig = plot_transit_vetting(self.ticid, self.lc, tpf)

        plt.show()

    def plot(self, outdir=None, lc_source='eleanor', input_lc=None, method=None, show=False, save_data=True, **kwargs):
        """Produce a quick look plot to characterize giants in the TESS catalog.

        Parameters
        ----------
        ticid : int
            TIC ID of desired target
        lc_source : "lightkurve" or "eleanor"
            Which package do you want to use to access the data?
        outdir : str
            Directory to save quick look plots into. Must be an existing directory.
        input_lc : ~lightkurve.LightCurve
            A LightCurve object containing an injection recovery test signal.

        Saves
        -----
        {tic}_quicklook.png : png image
            PNG of quick look plot
        """
        plt.clf()

        '''
        Plot Light Curve
        ----------------
        '''

        if outdir is None:
            outdir = os.path.join(self.PACKAGEDIR, 'outputs')

        plt.subplot2grid((4,4),(0,0),colspan=2)

        if lc_source == 'eleanor':
            lcc = self.from_eleanor(**kwargs)
            for lc, label, offset in zip(lcc, ['raw', 'corr'], [0, 0.01]):
                plt.plot(lc.time, lc.flux + offset, label=label)
            for val in self.breakpoints:
                plt.axvline(val, c='b', linestyle='dashed')
            plt.legend(loc=0)

            lc = lcc[1] # using corr_lc
            time = lc.time
            flux = lc.flux
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        elif lc_source == 'input':
            plt.plot(lc.time, lc.flux, label=lc.label)
            self.breakpoints = []
            time = lc.time
            flux = lc.flux
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        self.lc_exists = True

        lc = self._clean_data(lc)
        time, flux, flux_err = lc.time, lc.flux, lc.flux_err

        fig = plot_quicklook(lc, self.ticid, self.breakpoints, self.target_list, save_data, outdir)

        # save figure, timeseries, fft, and basic stats
        fig.savefig(str(outdir)+'/plots/'+str(self.ticid)+'_quicklook.png')
        if show:
            plt.show()

    def pdf_summary(self, outdir=None):
        """
        Generates a full PDF summary including vetting and transit model fitting.

        Parameters
        ----------
        outdir : path or str
            directory to store pdf output
        """
        if outdir is None:
            outdir = os.path.join(self.PACKAGEDIR, 'outputs')
        with PdfPages(os.path.join(outdir, str(self.ticid)+'_summary.pdf')) as pdf:

            ql_fig = self.plot(save_postcard=True, outdir=outdir)
            pdf.savefig(ql_fig)
            plt.close()
            lc = self.lc

            time, flux, flux_err = lc.time, lc.flux, lc.flux_err

            model = BoxLeastSquares(time, flux)
            results = model.autopower(0.16, minimum_period=2., maximum_period=21.)
            period = results.period[np.argmax(results.power)]
            t0 = results.transit_time[np.argmax(results.power)]

            vt_fig = plot_transit_vetting(f'TIC {self.ticid}', period, t0, lc=self.lc)
            pdf.savefig(vt_fig)
            plt.close()

            ica_fig = make_ica_plot(self.ticid)
            pdf.savefig(ica_fig)
            plt.close()

            star_fig = self.plot_starry_model(lc)
            pdf.savefig(star_fig)
            plt.close()

    def validate_transit(self, ticid=None, lc=None, rprs=0.02):
        """Take a closer look at potential transit signals."""
        from .plotting import create_starry_model

        if ticid is not None:
            lc = self.from_eleanor()[1]
            lc = self._clean_data(lc)
        elif lc is None:
            lc = self.lc

        model = BoxLeastSquares(lc.time, lc.flux)
        results = model.autopower(0.16)
        period = results.period[np.argmax(results.power)]
        t0 = results.transit_time[np.argmax(results.power)]
        if rprs is None:
            depth = results.depth[np.argmax(results.power)]
            rprs = depth ** 2

        # create the model
        model_flux = create_starry_model(lc.time, period=period, t0=t0, rprs=rprs) - 1
        model_lc = lk.LightCurve(time=lc.time, flux=model_flux)

        fig, ax = plt.subplots(3, 1, figsize=(12,14))
        fig.patch.set_facecolor('white')
        '''
        Plot unfolded transit
        ---------------------
        '''
        lc.scatter(ax=ax[0], c='k', label='Corrected Flux')
        model_lc.plot(ax=ax[0], c='r', lw=2, label='Transit Model')
        ax[0].set_ylim([-.002, .002])
        ax[0].set_xlim([lc.time[0], lc.time[-1]])

        '''
        Plot folded transit
        -------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[1], c='k', label=f'P={period:.3f}, t0={t0}')
        lc.fold(period, t0).bin(binsize=7).plot(ax=ax[1], c='b', label='binned', lw=2)
        model_lc.fold(period, t0).plot(ax=ax[1], c='r', lw=2, label="transit Model")
        ax[1].set_xlim([-0.5, .5])
        ax[1].set_ylim([-.002, .002])

        '''
        Zoom folded transit
        -------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[2], c='k', label=f'folded at {period:.3f} days')
        lc.fold(period, t0).bin(binsize=7).plot(ax=ax[2], c='b', label='binned', lw=2)
        model_lc.fold(period, t0).plot(ax=ax[2], c='r', lw=2, label="transit Model")
        ax[2].set_xlim([-0.1, .1])
        ax[2].set_ylim([-.002, .002])

        ax[0].set_title(f'{self.ticid}', fontsize=14)

        plt.show()

    def plot_gaia_overlay(self, ticid=None, tpf=None, cutout_size=9):
        """
        Check if the source is contaminated by displaying nearby Gaia targets as red circles.

        Parameters
        ----------
        ticid : int
            TIC ID of target
        tpf : `lightkurve.TargetPixelFile` object
            target pixel file on which to overplot Gaia stars
        cutout_size : int, tuple
            pixels on sides of tpf, tuples in form (y pixels, x pixels) accepted
        """
        from .plotting import add_gaia_figure_elements

        if ticid is None:
            ticid = self.ticid

        if tpf is None:
            tpf = get_cutout(ticid)

        fig = tpf.plot()
        fig = add_gaia_figure_elements(tpf, fig)

        return fig

    def _estimate_duration(self, p, rs, rp, b, a):
        """Hidden function to estimate duration given orbital parameters."""

        X = np.sqrt((rs + (rp*u.jupiterRad).to(u.solRad).value)**2 - b**2) / a
        td = (p / np.pi) * np.arcsin(X)

        return td

    def fit_starry_model(self, lc=None, **kwargs):
        """
        Generate and optimize a starry ocultation model.

        Parameters
        ----------
        lc : `lightkurve.LightCurve` object or None
            timeseries to fit model to. uses `Giant.lc` if None

        Returns
        -------
        model : `starry` model object
            full `starry` model output
        model_lc : `lightkurve.LightCurve` object
            light curve with transit model as flux time series
        """
        from .utils import _fit
        if lc is None:
            lc = self.lc

        x, y, yerr = lc.time, lc.flux, lc.flux_err
        model, static_lc = _fit(x, y, yerr, target_list=self.target_list, **kwargs)

        model_lc = lk.LightCurve(time=x, flux=static_lc)

        return model, model_lc

    def plot_starry_model(self, lc=None, model=None, **kwargs):
        """
        Generate and optimize a starry ocultation model, and plot.

        Parameters
        ----------
        lc : `lightkurve.LightCurve` object or None
            timeseries to fit model to. uses `Giant.lc` if None

        Returns
        -------
        fig : matplotlib.Figure
            figure object with transit fit subplots
        """
        print('Beginning starry fit...')
        if lc is None:
            lc = self.lc

        if model is None:
            model, model_lc = self.fit_starry_model(**kwargs)

        with model:
            period = model.map_soln['period'][0]
            t0 = model.map_soln['t0'][0]
            r_pl = model.map_soln['r_pl'] * 9.96
            a = model.map_soln['a'][0]
            b = model.map_soln['b'][0]

        try:
            r_star = self.target_list[self.target_list['ID'] == self.ticid]['rad'].values[0]
        except:
            r_star = 10.

        dur = self._estimate_duration(period, r_star, r_pl, b, a)

        fig, ax = plt.subplots(3, 1, figsize=(12,14))
        fig.patch.set_facecolor('white')
        '''
        Plot unfolded transit
        ---------------------
        '''
        lc.scatter(ax=ax[0], s=30, c='gray', alpha=.35, label='Corrected Flux')
        lc.bin(binsize=20).scatter(ax=ax[0], c='royalblue', s=50, alpha=.75, label='binned')
        model_lc.plot(ax=ax[0], c='C1', lw=2, label='Transit Model')
        ax[0].set_ylim([-.003, .002])
        ax[0].set_xlim([lc.time[0], lc.time[-1]])

        '''
        Plot folded transit
        -------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[1], s=30, c='gray', alpha=.35, label=rf'$P={period:.3f}, t0={t0:.3f}, '
                                                             rf'R_p={r_pl:.3f} R_J, b={b:.3f}, '
                                                             rf'\tau_T$={dur:.3f} days ({dur * 24:.3f} hrs)')
        lc.fold(period, t0).bin(binsize=20).scatter(ax=ax[1], c='royalblue', alpha=.75, s=50)
        model_lc.fold(period, t0).plot(ax=ax[1], c='C1', lw=2)
        ax[1].set_xlim([-0.5, .5])
        ax[1].set_ylim([-.003, .002])

        '''
        Zoom folded transit
        -------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[2], s=30, c='gray', alpha=.35, label=f'folded at {period:.3f} days')
        lc.fold(period, t0).bin(binsize=20).scatter(ax=ax[2], c='royalblue', alpha=.75, s=50)
        model_lc.fold(period, t0).plot(ax=ax[2], c='C1', lw=2, label="transit Model")
        ax[2].set_xlim([-0.1, 0.1])
        ax[2].set_ylim([-.003, .002])

        ax[0].set_title(f'{self.ticid}', fontsize=14)

        return fig

    def validate_ktransit(self, ticid=None, lc=None, rprs=0.02):
        """Take a closer look at the ktransit fit."""
        if ticid is not None:
            lc = self.from_eleanor()[1]
            lc = self._clean_data(lc)
        elif lc is None:
            lc = self.lc

        fitT = self.build_ktransit_model(ticid=ticid, lc=lc, rprs=rprs)

        res = fitT.fitresultplanets
        res2 = fitT.fitresultstellar

        fig = ktransit.plot_results(lc.time,lc.flux,fitT.transitmodel)

        fig.show()

    def simple_pca(self, tpf):
        """
        De-trending algorithm for `lightkurve` version of FFI pipeline.

        The steps of this de-trending are:
         - Find a threshold aperture mask around the target
         - Create a design matrix with column vectors from pixel-level timeseries outside of the aperture
         - Perform Principle Component Analysis (PCA) on column vectors to find out background model vectors
         - Fit weights to these vectors to minimize squared difference between model and observations
         - Subtract noise model

        Parameters
        ----------
        tpf : `lightkurve.TargetPixelFile` object
            target pixel file for desired target

        Returns
        -------
        corrected_lc : `lightkurve.LightCurve` object
            background-corrected light curve
        """
        aper = tpf._parse_aperture_mask('threshold')
        raw_lc = tpf.to_lightcurve(aperture_mask=aper).remove_nans()
        mask = raw_lc.flux_err > 0
        self.raw_lc = raw_lc[mask]
        tpf = tpf[mask]


        regressors = tpf.flux[:, ~aper]

        dm = lk.DesignMatrix(regressors, name='regressors')

        dm = dm.pca(10)
        dm = dm.append_constant()

        corrector = lk.RegressionCorrector(self.raw_lc.normalize())
        corrected_lc = corrector.correct(dm)

        return corrected_lc

    def save_to_fits(self, outdir=None):
        """
        Pipeline to download and de-trend a target using the `lightkurve` implememtation.
        Downloads data, removes background, and saves as fits files. This function outputs:
         - {TICID}_s{SECTOR}_corr.fits : corrected light curve
         - {TICID}_s{SECTOR}_raw.fits : raw SAP flux light curve

        Parameters
        ----------
        outdir : str or path
            location of fits output
        """
        self.silent = True

        if outdir is None:
            outdir = os.path.join(self.PACKAGEDIR, 'outputs')

        sectors = self._find_sectors(self.ticid)

        for s in sectors:
            self.fetch_and_clean_data(lc_source='lightkurve', sectors=s, gauss_filter_lc=False)

            fname_corr = f'{self.ticid}_s{s:02d}_corr.fits'
            fname_raw = f'{self.ticid}_s{s:02d}_raw.fits'

            path_corr = os.path.join(outdir, fname_corr)
            path_raw = os.path.join(outdir, fname_raw)

            self.lc.flux += 1.

            self.lc.to_fits(path=path_corr, overwrite=True)
            self.raw_lc.to_fits(path=path_raw, overwrite=True)



if __name__ == '__main__':
    try:
        ticid = sys.argv[1]
        outdir = sys.argv[2]
        output = "plot"
        try:
            output = sys.argv[3]
        except:
            pass

        target = Giant(ticid=ticid, csv_path='data/ticgiants_allsky_halo.csv', cache_path='/data/sarek1/nksaun/lightkurve_cache')

        if output=="plot":
            target.fetch_and_clean_data(lc_source='lightkurve')
            plot_summary(target, outdir=outdir, save_data=True)
        else:
            target.save_to_fits(outdir=outdir)
    except:
        print(f'Target {sys.argv[1]} failed.')
