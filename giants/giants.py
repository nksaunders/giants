import os
import re
import eleanor
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares
import astropy.stats as ass
import lightkurve as lk
from . import PACKAGEDIR
import warnings
import astropy.stats as ass
# suppress verbose astropy warnings and future warnings
warnings.filterwarnings("ignore", module="astropy")
warnings.filterwarnings("ignore", category=FutureWarning)

class Giant(object):
    """An object to store and analyze time series data for giant stars.
    """
    def __init__(self):
        self.cvz = self.get_cvz_targets()
        self.brightcvz = self.cvz.GAIAmag < 6.5
        # print(f'Using the brightest {len(self.cvz[self.brightcvz])} targets.')

    def get_cvz_targets(self):
        """Read in a csv of CVZ targets from a local file.
        """
        try:
            # full list
            path = os.path.abspath(os.path.join(PACKAGEDIR, 'data', 'TICgiants_bright.csv'))
        except:
            # shorter list
            path = os.path.abspath(os.path.join(PACKAGEDIR, 'data', 'TICgiants_CVZ.csv'))
        return pd.read_csv(path, skiprows=4)

    def get_target_list(self):
        """Helper function to fetch a list of TIC IDs.

        Returns
        -------
        IDs : list
            TIC IDs for bright targets in list.
        """
        return self.cvz[self.brightcvz].ID.values

    def from_lightkurve(self, ind=0, ticid=None, pld=True, cutout_size=5):
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
            ticid = self.cvz[self.brightcvz].ID.values[i]
        # search TESScut for the desired target, read its sectors
        sr = lk.search_tesscut('tic{}'.format(ticid))
        sectors = self._find_sectors(sr)
        print(f'Creating light curve for target {ticid} for sectors {sectors}.')
        # download the TargetPixelFileCollection for TESScut observations
        tpfc = sr.download_all(cutout_size=cutout_size)
        rlc = self._photometry(tpfc[0], pld=False).normalize()
        # track breakpoints between sectors
        self.breakpoints = [rlc.time[-1]]
        # iterate through TPFs and perform photometry on each of them
        for t in tpfc[1:]:
            single_rlc = self._photometry(t, pld=False).normalize()
            rlc = rlc.append(single_rlc)
            self.breakpoints.append(single_rlc.time[-1])
        rlc.label = 'Raw {ticid}'
        # do the same but with de-trending (if you want)
        if pld:
            clc = self._photometry(tpfc[0], pld=True).normalize()
            for t in tpfc[1:]:
                single_clc = self._photometry(t, pld=True).normalize()
                clc = clc.append(single_clc)
            clc.label = 'PLD {ticid}'
            rlc = rlc.remove_nans()
            clc = clc.remove_nans()
            return lk.LightCurveCollection([rlc, clc])
        else:
            rlc = rlc.remove_nans()
            return lk.LightCurveCollection([rlc])

    def from_eleanor(self, ticid):
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
        # search TESScut to figure out which sectors you need (there's probably a better way to do this)
        sr = lk.search_tesscut(ticid)
        sectors = self._find_sectors(sr)
        if isinstance(ticid, str):
            ticid = int(re.search(r'\d+', str(ticid)).group())
        print(f'Creating light curve for target {ticid} for sectors {sectors}.')
        # download target data for the desired source for only the first available sector
        star = eleanor.Source(tic=ticid, sector=sectors[0], tc=True)
        data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True)
        q = data.quality == 0
        # create raw flux light curve
        raw_lc = lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q],label='raw', time_format='btjd').remove_nans().normalize()
        corr_lc = lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], label='corr', time_format='btjd').remove_nans().normalize()
        pca_lc = lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], flux_err=data.flux_err[q],label='pca', time_format='btjd').remove_nans().normalize()
        psf_lc = lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], flux_err=data.flux_err[q],label='psf', time_format='btjd').remove_nans().normalize()
        #track breakpoints between sectors
        self.breakpoints = [raw_lc.time[-1]]
        # iterate through extra sectors and append the light curves
        if len(sectors) > 1:
            for s in sectors[1:]:
                star = eleanor.Source(tic=ticid, sector=s, tc=True)
                data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True)
                q = data.quality == 0

                raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                pca_lc = pca_lc.append(lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                psf_lc = psf_lc.append(lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())

                self.breakpoints.append(raw_lc.time[-1])
        # store in a LightCurveCollection object and return
        return lk.LightCurveCollection([raw_lc, corr_lc, pca_lc, psf_lc])

    def _find_sectors(self, sr):
        """Helper function to read sectors from a search result."""
        sectors = []
        for desc in sr.table['description']:
            sectors.append(int(re.search(r'\d+', str(desc)).group()))
        return sectors

    def _photometry(self, tpf, pld=True):
        """Helper function to perform photometry on a pixel level observation."""
        if pld:
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
        burnin[:40] = True
        # also 6 sigma outliers
        _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        mask = momdump | outliers | burnin
        lc.time = lc.time[~mask]
        lc.flux = lc.flux[~mask]
        lc.flux_err = lc.flux_err[~mask]
        lc.flux = lc.flux - 1
        lc.flux = lc.flux - scipy.ndimage.filters.gaussian_filter(lc.flux, 100)

        # store cleaned lc
        self.lc = lc
        return lc

    def plot(self, ticid, lc_source='eleanor', outdir='plots', input_lc=None):
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
        self.ticid = ticid
        plt.subplot2grid((4,4),(0,0),colspan=2)

        if lc_source == 'lightkurve':
            lcc = self.from_lightkurve(ticid)
            q = lcc[0].quality == 0

            plt.plot(lcc[0].time[q], lcc[0].flux[q], 'k', label="Raw")
            if len(lcc) > 1:
                plt.plot(lcc[1].time[q], lcc[1].flux[q], 'r', label="Corr")
            for val in self.breakpoints:
                plt.axvline(val, c='b', linestyle='dashed')
            lc = lcc[-1]
            time = lc.time[q]
            flux = lc.flux[q] # - 1
            # flux = flux - scipy.ndimage.filters.gaussian_filter(flux, 100)
            flux_err = lc.flux_err[q]
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        elif lc_source == 'eleanor':
            lcc = self.from_eleanor(ticid)
            for lc, label, offset in zip(lcc, ['raw', 'corr', 'pca', 'psf'], [-0.1, 0, 0.1, -.2]):
                plt.plot(lc.time, lc.flux + offset, label=label)
            for val in self.breakpoints:
                plt.axvline(val, c='b', linestyle='dashed')
            plt.legend(loc=0)

            lc = lcc[1] # using corr_lc
            time = lc.time
            flux = lc.flux # - 1
            # flux = flux - scipy.ndimage.filters.gaussian_filter(flux, 100)
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        elif lc_source == 'input':
            lc = input_lc
            plt.plot(lc.time, lc.flux, label=lc.label)
            self.breakpoints = []
            time = lc.time
            flux = lc.flux # - 1
            # flux = flux - scipy.ndimage.filters.gaussian_filter(flux, 100)
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        lc = self._clean_data(lc)
        time, flux, flux_err = lc.time, lc.flux, lc.flux_err

        """    # mask first 12h after momentum dump
        momdump = (time > 1339) * (time < 1341)
        # also the burn in
        burnin = np.zeros_like(time, dtype=bool)
        burnin[:120] = True
        # also 6 sigma outliers
        _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        mask = momdump | burnin | outliers
        time = time[~mask]
        flux = flux[~mask]
        flux_err = flux_err[~mask]

        # store masked values
        self.lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)"""

        '''
        Plot Filtered Light Curve
        -------------------------
        '''
        plt.subplot2grid((4,4),(1,0),colspan=2)

        plt.plot(time, flux, 'k', label="filtered")
        for val in self.breakpoints:
            plt.axvline(val, c='b', linestyle='dashed')
        plt.legend()
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')

        # power = lc.to_periodogram().power
        freq = np.linspace(1./15, 1./.01, 100000)
        power = ass.LombScargle(time, flux, flux_err).power(freq)
        ps = 1./freq

        '''
        Plot Periodogram
        ----------------
        '''
        plt.subplot2grid((4,4),(0,2),colspan=2,rowspan=4)
        plt.loglog(freq/24/3600 * 1e6, power)
        plt.loglog(freq/24/3600 * 1e6, scipy.ndimage.filters.gaussian_filter(power, 50), color='r', alpha=0.8, lw=2.5)
        plt.axvline(283,-1,1, ls='--', color='k')
        plt.xlabel("Frequency [uHz]")
        plt.ylabel("Power")
        plt.xlim(10, 400)
        try:
            # annotate with stellar params
            # won't work for TIC ID's not in the list
            if isinstance(ticid, str):
                ticid = int(re.search(r'\d+', str(ticid)).group())
            Gmag = self.cvz[self.cvz['ID'] == ticid]['GAIAmag'].values[0]
            Teff = self.cvz[self.cvz['ID'] == ticid]['Teff'].values[0]
            R = self.cvz[self.cvz['ID'] == ticid]['rad'].values[0]
            M = self.cvz[self.cvz['ID'] == ticid]['mass'].values[0]
            plt.annotate(rf"G mag = {Gmag:.3f}", xy=(.05, .08), xycoords='axes fraction')
            plt.annotate(rf"Teff = {int(Teff)} K", xy=(.05, .06), xycoords='axes fraction')
            plt.annotate(rf"R = {R:.3f} $R_\odot$", xy=(.05, .04), xycoords='axes fraction')
            plt.annotate(rf"M = {M:.3f} $M_\odot$", xy=(.05, .02), xycoords='axes fraction')
        except:
            pass

        '''
        Plot BLS
        --------
        '''
        plt.subplot2grid((4,4),(2,0),colspan=2)

        model = BoxLeastSquares(time, flux)
        results = model.autopower(0.16, maximum_period=27.)
        period = results.period[np.argmax(results.power)]
        t0 = results.transit_time[np.argmax(results.power)]

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
        plt.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes,40), lw=2, color='r', label=f'P={period:.2f} days')
        plt.xlabel('Phase')
        plt.ylabel('Flux')
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.0025, 0.0025)
        plt.legend(loc=0)

        fig = plt.gcf()
        fig.suptitle(f'{ticid}', fontsize=14)
        fig.set_size_inches(12, 10)

        fig.savefig(outdir+'/'+str(ticid)+'_quicklook.png')
        plt.show()

    def validate_transit(self, ticid=None, lc=None, rprs=0.02):
        """Take a closer look at potential transit signals."""
        from .utils import create_starry_model

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

    def plot_gaia_overlay(self, ticid=None, tpf=None):
        """Check if the source is contaminated."""
        from .utils import add_gaia_figure_elements

        if ticid is None:
            ticid = self.ticid

        if tpf is None:
            tpf = lk.search_tesscut(ticid)[0].download(cutout_size=9)

        fig = tpf.plot()
        fig = add_gaia_figure_elements(tpf, fig)

        return fig
