import os
import re
import eleanor
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares
from sklearn.decomposition import FastICA, PCA
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
    def __init__(self, csv_path='data/ticgiants_bright_v2_skgrunblatt.csv'):
        self.cvz = self.get_cvz_targets(csv_path)
        self.brightcvz = np.ones(len(self.cvz), dtype=bool) # self.cvz.GAIAmag < 12.5
        # print(f'Using the brightest {len(self.cvz[self.brightcvz])} targets.')

    def get_cvz_targets(self, csv_path='data/ticgiants_bright_v2_skgrunblatt.csv'):
        """Read in a csv of CVZ targets from a local file.
        """
        path = os.path.abspath(os.path.abspath(os.path.join(PACKAGEDIR, csv_path)))
        return pd.read_csv(path, skiprows=0)

    def get_target_list(self):
        """Helper function to fetch a list of TIC IDs.

        Returns
        -------
        IDs : list
            TIC IDs for bright targets in list.
        """
        return self.cvz[self.brightcvz].ID.values

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
            ticid = self.cvz[self.brightcvz].ID.values[i]
        # search TESScut for the desired target, read its sectors
        sr = lk.search_tesscut(ticid)
        sectors = self._find_sectors(sr)
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
                try: # some sectors fail randomly
                    star = eleanor.Source(tic=ticid, sector=s, tc=True)
                    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True)
                    q = data.quality == 0

                    raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    pca_lc = pca_lc.append(lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    psf_lc = psf_lc.append(lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())

                    self.breakpoints.append(raw_lc.time[-1])
                except:
                    continue
        # store in a LightCurveCollection object and return
        return lk.LightCurveCollection([raw_lc, corr_lc, pca_lc, psf_lc])

    def _find_sectors(self, sr):
        """Helper function to read sectors from a search result."""
        sectors = []
        for desc in sr.table['description']:
            sectors.append(int(re.search(r'\d+', str(desc)).group()))
        return sectors

    def _photometry(self, tpf, method=None, use_gp=False):
        """Helper function to perform photometry on a pixel level observation."""
        if method=='pld':
            pld = tpf.to_corrector('pld')
            lc = pld.correct(aperture_mask='threshold', pld_aperture_mask='all', use_gp=False)
        elif method=='ica':
            n_components = 20
            flux = tpf.flux
            pixmask = tpf.create_threshold_mask()

            # ica = FastICA(n_components=n_components, tol=0.1, max_iter=500)
            # X = ica.fit_transform(flux[:,~pixmask].reshape(len(flux[:,~pixmask]), -1))
            from fbpca import pca
            X, _, _ = pca(flux[:,~pixmask], 20, n_iter=10)


            lc = tpf.to_lightcurve(aperture_mask=pixmask)
            ivar = 1.0 / lc.flux_err**2 # inverse variance

            # XTX = np.dot(X.T, X * ivar[:, None])
            # XTy = np.dot(X.T, lc.flux * ivar)
            XTX = np.dot(X.T, X)
            XTy = np.dot(X.T, lc.flux)
            w = np.linalg.solve(XTX, XTy)
            m = np.dot(X, w)
            '''
            if use_gp:
                y = lc.flux - m
                amp = np.nanstd(y)
                tau = 30
                kernel = celerite.terms.Matern32Term(np.log(amp), np.log(tau))
                gp = celerite.GP(kernel)
                gp.compute(lc.time, lc.flux_err)

                # compute the coefficients C on the basis vectors;
                # the PLD design matrix will be dotted with C to solve for the noise model.
                XTX = np.dot(X.T, gp.apply_inverse(X))
                XTy = np.dot(X.T, gp.apply_inverse(lc.flux[:, None])[:, 0])
                w = np.linalg.solve(XTX, XTy)
                m = np.dot(X, w)
            '''

            lc.flux = lc.flux - m
            return lc
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

    def plot(self, ticid, lc_source='eleanor', outdir='plots', input_lc=None, method=None):
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
            lcc = self.from_eleanor(ticid)
            for lc, label, offset in zip(lcc, ['raw', 'corr', 'pca', 'psf'], [-0.1, 0, 0.1, -.2]):
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

        lc = self._clean_data(lc)
        time, flux, flux_err = lc.time, lc.flux, lc.flux_err

        model = BoxLeastSquares(time, flux)
        results = model.autopower(0.16, minimum_period=1., maximum_period=30.)
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
        for val in self.breakpoints:
            plt.axvline(val, c='b', linestyle='dashed')
        plt.legend()
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')

        # power = lc.to_periodogram().power
        freq = np.linspace(1./15, 1./.01, 100000)
        power = lc.to_periodogram('lombscargle', frequency=freq).power
        #power = ass.LombScargle(time, flux, flux_err).power(freq)
        ps = 1./freq

        '''
        Plot Periodogram
        ----------------
        '''
        plt.subplot2grid((4,4),(0,2),colspan=2,rowspan=4)
        plt.loglog(freq/24/3600 * 1e6, power)
        plt.loglog(freq/24/3600 * 1e6, scipy.ndimage.filters.gaussian_filter(power, 150), color='r', alpha=0.8, lw=2.5)
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
            plt.annotate(rf"G mag = {Gmag:.3f}", xy=(.02, .08), xycoords='axes fraction')
            plt.annotate(rf"Teff = {int(Teff)} K", xy=(.02, .06), xycoords='axes fraction')
            plt.annotate(rf"R = {R:.3f} $R_\odot$", xy=(.02, .04), xycoords='axes fraction')
            plt.annotate(rf"M = {M:.3f} $M_\odot$", xy=(.02, .02), xycoords='axes fraction')
        except:
            pass
        plt.annotate(f'depth = {depth:.4f}', xy=(.35, .08), xycoords='axes fraction')
        plt.annotate(f'depth_snr = {depth_snr:.4f}', xy=(.35, .06), xycoords='axes fraction')
        plt.annotate(f'period = {period:.3f} days', xy=(.35, .04), xycoords='axes fraction')
        plt.annotate(f't0 = {t0:.3f}', xy=(.35, .02), xycoords='axes fraction')

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
        plt.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes, 40), lw=2, color='r', label=f'P={period:.2f} days')
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

    def fit_starry_model(self, lc=None, **kwargs):
        """

        """
        if lc is None:
            lc = self.lc

        x, y, yerr = lc.time, lc.flux, lc.flux_err
        model, static_lc = self._fit(x, y, yerr, **kwargs)

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
        lc.fold(period, t0).scatter(ax=ax[1], c='k', label=rf'$P={period:.3f}, t0={t0:.3f}, R_p={r_pl:.3f} R_J$')
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
        ax[2].set_xlim([-0.05, 0.05])
        ax[2].set_ylim([-.002, .002])

        ax[0].set_title(f'{self.ticid}', fontsize=14)

        plt.show()

    def _fit(self, x, y, yerr, period_prior=None, t0_prior=None, depth=None, **kwargs):
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

        # build_model should only take lc and system
        # model = build_model(lc, system)

        def build_model(x, y, yerr, period_prior, t0_prior, depth, minimum_period=3, maximum_period=30, r_star_prior=5.0, t_star_prior=5000, rho_star_prior=0.07, start=None):
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
            try:
                import exoplanet as xo
                import pymc3 as pm
                import theano.tensor as tt

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

                # Fixed stellar parameters (isochrones with JHK + parallax)
                # m_star = pm.Normal("m_star", mu=1.7332613382990079, sd=0.07346535475676119)
                r_star = pm.Normal("r_star", mu=r_star_prior, sd=2.0)
                t_star = pm.Normal("t_star", mu=t_star_prior, sd=200)
                rho_star = pm.Normal("rho_star", mu=rho_star_prior, sd=.1)

                '''Orbital Parameters'''
                # The time of a reference transit for each planet
                t0 = pm.Normal("t0", mu=t0_prior, sd=5., shape=1)
                period = pm.Uniform("period", testval=period_prior,
                                    lower=period_prior+(-5.),
                                    upper=period_prior+(5.),
                                    shape=1)
                # logP = pm.Normal("logP", mu=np.log(period_prior), sd=.2)
                # Tracking planet parameters
                # period = pm.Deterministic("period", tt.exp(logP))
                b = pm.Uniform("b", testval=0.5, shape=1)

                # Set up a Keplerian orbit for the planets
                model.orbit = xo.orbits.KeplerianOrbit(
                    period=period, t0=t0, b=b, r_star=r_star, rho_star=rho_star)# m_star=m_star)

                # track additional orbital parameters
                a = pm.Deterministic("a", model.orbit.a)
                incl = pm.Deterministic("incl", model.orbit.incl)

                '''Planet Parameters'''
                # quadratic limb darkening paramters
                u = xo.distributions.QuadLimbDark("u")

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
                model.light_curves = xo.StarryLightCurve(u).get_light_curve(
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
