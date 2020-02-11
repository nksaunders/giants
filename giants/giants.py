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
from matplotlib.backends.backend_pdf import PdfPages
try:
    from . import lomb
except:
    import lomb

from .plotting import plot_quicklook, plot_transit_vetting, make_ica_plot, superplot
from .utils import get_cutout

# suppress verbose astropy warnings and future warnings
warnings.filterwarnings("ignore", module="astropy")
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ['Giant']

class Giant(object):
    """An object to store and analyze time series data for giant stars.
    """
    def __init__(self, csv_path='data/ticgiants_bright_v2_skgrunblatt.csv'):
        self.target_list = self.get_targets(csv_path)
        self.lc_exists = False

    def get_targets(self, csv_path='data/ticgiants_bright_v2_skgrunblatt.csv'):
        """Read in a csv of CVZ targets from a local file.
        """
        self.PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
        path = os.path.abspath(os.path.abspath(os.path.join(self.PACKAGEDIR, csv_path)))
        return pd.read_csv(path, skiprows=0, dtype='unicode')

    def get_target_list(self):
        """Helper function to fetch a list of TIC IDs.

        Returns
        -------
        IDs : list
            TIC IDs for bright targets in list.
        """
        return self.target_list.ID.values

    def from_lightkurve(self, ind=0, ticid=None, method=None, cutout_size=9):
        """Download cutouts around target for each sector using Lightkurve
        and create light curves.
        Requires either `ind` or `ticid`.

        Parameters
        ----------
        ind : int
            Index of target array to download files for
        ticid : int
            TIC ID of desired target
        pld : boolean
            Option to detrend raw light curve with PLD
        cutout_size : int or tuple
            Dimensions of TESScut cutout in pixels

        Returns
        -------
        LightCurveCollection :
            ~lightkurve.LightCurveCollection containing raw and corrected light curves.
        """
        if ticid == None:
            i = ind
            ticid = self.target_list.ID.values[i]
        # search TESScut for the desired target, read its sectors
        sectors = self._find_sectors(ticid)
        if isinstance(ticid, str):
            ticid = int(re.search(r'\d+', str(ticid)).group())
        print(f'Creating light curve for target {ticid} for sectors {sectors}.')
        # download the TargetPixelFileCollection for TESScut observations
        tpfc = sr.download_all(cutout_size=cutout_size)
        rlc = self._photometry(tpfc[0]).normalize()
        # track breakpoints between sectors
        self.breakpoints = [rlc.time[-1]]
        # iterate through TPFs and perform photometry on each of them
        for t in tpfc[1:]:
            single_rlc = self._photometry(t).normalize()
            rlc = rlc.append(single_rlc)
            self.breakpoints.append(single_rlc.time[-1])
        rlc.label = 'Raw {ticid}'
        # do the same but with de-trending (if you want)
        if method is not None:
            clc = self._photometry(tpfc[0], method=method).normalize()
            for t in tpfc[1:]:
                single_clc = self._photometry(t, method=method).normalize()
                clc = clc.append(single_clc)
            clc.label = 'PLD {ticid}'
            rlc = rlc.remove_nans()
            clc = clc.remove_nans()
            return lk.LightCurveCollection([rlc, clc])
        else:
            rlc = rlc.remove_nans()
            return lk.LightCurveCollection([rlc])

    def from_eleanor(self, ticid, save_postcard=False):
        """Download light curves from Eleanor for desired target. Eleanor light
        curves include:
        - raw : raw flux light curve
        - corr : corrected flux light curve
        - pca : principle component analysis light curve
        - psf : point spread function photometry light curve

        Parameters
        ----------
        ticid : int
            TIC ID of desired target

        Returns
        -------
        LightCurveCollection :
            ~lightkurve.LightCurveCollection containing raw and corrected light curves.
        """
        '''
        # BUGFIX FOR ELEANOR (DEPRICATED)
        # -------------------------------
        from astroquery.mast import Observations
        server = 'https://mast.stsci.edu'
        Observations._MAST_REQUEST_URL = server + "/api/v0/invoke"
        Observations._MAST_DOWNLOAD_URL = server + "/api/v0.1/Download/file"
        Observations._COLUMNS_CONFIG_URL = server + "/portal/Mashup/Mashup.asmx/columnsconfig"
        '''

        # search TESScut to figure out which sectors you need (there's probably a better way to do this)
        if isinstance(ticid, str):
            ticid = int(re.search(r'\d+', str(ticid)).group())
        self.ticid = ticid
        sectors = self._find_sectors(f'TIC {ticid}')
        print(f'Creating light curve for target {ticid} for sectors {sectors}.')
        # download target data for the desired source for only the first available sector

        star = eleanor.Source(tic=ticid, sector=int(sectors[0]), tc=True)
        try:
            data = eleanor.TargetData(star, height=11, width=11, bkg_size=27, do_psf=False, do_pca=False, try_load=True, save_postcard=save_postcard)
        except:
            data = eleanor.TargetData(star, height=7, width=7, bkg_size=21, do_psf=False, do_pca=False, try_load=True, save_postcard=save_postcard)
        q = data.quality == 0
        # create raw flux light curve
        raw_lc = lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q],label='raw', time_format='btjd').remove_nans().normalize()
        corr_lc = lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], label='corr', time_format='btjd').remove_nans().normalize()
        # pca_lc = lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], flux_err=data.flux_err[q],label='pca', time_format='btjd').remove_nans().normalize()
        # psf_lc = lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], flux_err=data.flux_err[q],label='psf', time_format='btjd').remove_nans().normalize()
        #track breakpoints between sectors
        self.breakpoints = [raw_lc.time[-1]]
        # iterate through extra sectors and append the light curves
        if len(sectors) > 1:
            for s in sectors[1:]:
                try: # some sectors fail randomly
                    star = eleanor.Source(tic=ticid, sector=int(s), tc=True)
                    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False, try_load=True)
                    q = data.quality == 0

                    raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    # pca_lc = pca_lc.append(lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    # psf_lc = psf_lc.append(lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())

                    self.breakpoints.append(raw_lc.time[-1])
                except:
                    continue
        # store in a LightCurveCollection object and return
        return lk.LightCurveCollection([raw_lc, corr_lc])

    def _find_sectors(self, ticid):
        """Helper function to read sectors from a search result."""
        from astroquery.mast import Tesscut
        sectors = []
        for s in Tesscut().get_sectors(objectname=ticid)['sector']:
            sectors.append(s)
        return sectors

    def _photometry(self, tpf, method=None, use_gp=False):
        """Helper function to perform photometry on a pixel level observation."""
        if method=='pld':
            pld = tpf.to_corrector('pld')
            lc = pld.correct(aperture_mask='threshold', pld_aperture_mask='all', use_gp=False)
        else:
            lc = tpf.to_lightcurve(aperture_mask='threshold')
        return lc

    def _clean_data(self, lc):
        """ """
        # mask first 12h after momentum dump
        momdump = (lc.time > 1339) * (lc.time < 1341)
        # also the burn in
        burnin = np.zeros_like(lc.time, dtype=bool)
        burnin[:30] = True
        downlinks = [1339.6770629882812, 1368.6353149414062, 1421.239501953125, 1451.5728759765625, 1478.114501953125,
                     1504.7199096679688, 1530.2824096679688, 1535.0115966796875, 1556.74072265625, 1582.7824096679688,
                     1610.8031616210938, 1640.0531616210938, 1668.6415405273438, 1697.3673095703125, 1724.9667358398438,
                     1751.6751098632812]
        # mask around downlinks
        for d in downlinks:
            if d in lc.time:
                burnin[d:d+15] = True
        # also 6 sigma outliers
        _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        mask = momdump | outliers | burnin
        lc.time = lc.time[~mask]
        lc.flux = lc.flux[~mask]
        lc.flux_err = lc.flux_err[~mask]
        lc.flux = lc.flux - 1
        lc.flux = lc.flux - scipy.ndimage.filters.gaussian_filter(lc.flux, 90) # <2-day (5muHz) filter

        # store cleaned lc
        self.lc = lc
        return lc

    def vet_transit(self, ticid, lc=None, tpf=None, **kwargs):
        """

        """
        if not self.lc_exists:
            lcc = self.from_eleanor(ticid, **kwargs)
            lc = lcc[1] # using corr_lc
            time = lc.time
            flux = lc.flux
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
            self.lc = self._clean_data(lc)

        fig = plot_transit_vetting(ticid, self.lc, tpf)

        plt.show()

    def plot(self, ticid, outdir=None, lc_source='eleanor', input_lc=None, method=None, show=False, save_data=True, **kwargs):
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

        self.ticid = ticid
        plt.subplot2grid((4,4),(0,0),colspan=2)

        if lc_source == 'lightkurve':
            lcc = self.from_lightkurve(ticid=ticid, method=method)
            q = lcc[0].quality == 0

            plt.plot(lcc[0].time[q], lcc[0].flux[q], 'k', label="Raw")
            if len(lcc) > 1:
                q = lcc[1].quality == 0
                plt.plot(lcc[1].time[q], lcc[1].flux[q]+.2, 'r', label="Corr")
            for val in self.breakpoints:
                plt.axvline(val, c='b', linestyle='dashed')
            plt.legend(loc=0)
            lc = lcc[-1]
            time = lc.time[q]
            flux = lc.flux[q]
            flux_err = lc.flux_err[q]
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err).remove_nans()

        elif lc_source == 'eleanor':
            lcc = self.from_eleanor(ticid, **kwargs)
            for lc, label, offset in zip(lcc, ['raw', 'corr', 'pca', 'psf'], [-0.01, 0, 0.01, -.02]):
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

        fig = plot_quicklook(lc, ticid, self.breakpoints, self.target_list, save_data, outdir)

        # save figure, timeseries, fft, and basic stats
        fig.savefig(outdir+'/plots/'+str(ticid)+'_quicklook.png')
        if show:
            plt.show()

    def pdf_summary(self, ticid, out_fname):
        """

        """
        self.ticid=ticid
        with PdfPages(out_fname) as pdf:

            ql_fig = self.plot(self.ticid, save_postcard=True)
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
            lc = self.from_eleanor(ticid)[1]
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

        ax[0].set_title(f'{ticid}', fontsize=14)

        plt.show()

    def plot_gaia_overlay(self, ticid=None, tpf=None, cutout_size=9):
        """Check if the source is contaminated."""
        from .plotting import add_gaia_figure_elements

        if ticid is None:
            ticid = self.ticid

        if tpf is None:
            tpf = get_cutout(ticid)

        fig = tpf.plot()
        fig = add_gaia_figure_elements(tpf, fig)

        return fig

    def _estimate_duration(self, p, rs, rp, b, a):
        """ """

        X = np.sqrt((rs + (rp*u.jupiterRad).to(u.solRad).value)**2 - b**2) / a
        td = (p / np.pi) * np.arcsin(X)

        return td

    def fit_starry_model(self, lc=None, **kwargs):
        """

        """
        from .utils import _fit
        if lc is None:
            lc = self.lc

        x, y, yerr = lc.time, lc.flux, lc.flux_err
        model, static_lc = _fit(x, y, yerr, target_list=self.target_list, **kwargs)

        model_lc = lk.LightCurve(time=x, flux=static_lc)

        return model, model_lc

    def plot_starry_model(self, lc=None, model=None, **kwargs):
        """ """
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
        lc.scatter(ax=ax[0], c='k', label='Corrected Flux')
        lc.bin(binsize=7).plot(ax=ax[0], c='b', lw=1.5, alpha=.75, label='binned')
        model_lc.plot(ax=ax[0], c='r', lw=2, label='Transit Model')
        ax[0].set_ylim([-.002, .002])
        ax[0].set_xlim([lc.time[0], lc.time[-1]])

        '''
        Plot folded transit
        -------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[1], c='k', label=rf'$P={period:.3f}, t0={t0:.3f}, '
                                                             'R_p={r_pl:.3f} R_J, b={b:.3f}, '
                                                             '\tau_T$={dur:.3f} days ({dur * 24:.3f} hrs)')
        lc.fold(period, t0).bin(binsize=7).plot(ax=ax[1], c='b', alpha=.75, lw=2)
        model_lc.fold(period, t0).plot(ax=ax[1], c='r', lw=2)
        ax[1].set_xlim([-0.5, .5])
        ax[1].set_ylim([-.002, .002])

        '''
        Zoom folded transit
        -------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[2], c='k', label=f'folded at {period:.3f} days')
        lc.fold(period, t0).bin(binsize=7).plot(ax=ax[2], c='b', alpha=.75, lw=2)
        model_lc.fold(period, t0).plot(ax=ax[2], c='r', lw=2, label="transit Model")
        ax[2].set_xlim([-0.1, 0.1])
        ax[2].set_ylim([-.0015, .0015])

        ax[0].set_title(f'{self.ticid}', fontsize=14)

        return fig

    def validate_ktransit(self, ticid=None, lc=None, rprs=0.02):
        """ """
        if ticid is not None:
            lc = self.from_eleanor(ticid)[1]
            lc = self._clean_data(lc)
        elif lc is None:
            lc = self.lc

        fitT = self.build_ktransit_model(ticid=ticid, lc=lc, rprs=rprs)

        fitT.print_results()            # print some results
        res = fitT.fitresultplanets
        res2 = fitT.fitresultstellar

        fig = ktransit.plot_results(lc.time,lc.flux,fitT.transitmodel)

        fig.show()

    def make_superplot(self, ticid, save_data=False, outdir='', **kwargs):
        """

        """
        plt.clf()
        fig = plt.figure(figsize=(16,8))

        self.ticid = ticid
        plt.subplot2grid((8,16),(0,0),colspan=4, rowspan=1)
        lcc = self.from_eleanor(ticid, **kwargs)
        for lc, label, offset in zip(lcc, ['raw', 'corr', 'pca', 'psf'], [-0.01, 0, 0.01, -.02]):
            plt.plot(lc.time, lc.flux + offset, label=label)
        for val in self.breakpoints:
            plt.axvline(val, c='b', linestyle='dashed')
        plt.legend(loc=0)

        lc = lcc[1] # using corr_lc
        time = lc.time
        flux = lc.flux
        flux_err = np.ones_like(flux) * 1e-5
        lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        out_fig = superplot(lc, ticid, self.breakpoints, self.target_list, save_data, outdir)

        plt.show()


if __name__ == '__main__':
    try:
        target = Giant(csv_path='data/ticgiants_allsky_halo.csv')
        target.plot(*sys.argv[1:])
    except:
        print(f'No data found for target {sys.argv[1]}')
