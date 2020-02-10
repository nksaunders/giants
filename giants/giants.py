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
from multiprocessing import Pool
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
        if ticid == 'TIC 467438786':
                sectors = [16]
        if isinstance(ticid, str):
            ticid = int(re.search(r'\d+', str(ticid)).group())
        print(f'Creating light curve for target {ticid} for sectors {sectors}.')
        # download target data for the desired source for only the first available sector
        try:
            star = eleanor.Source(tic=ticid, sector=sectors[0], tc=True)
        except:
            star = eleanor.Source(tic=ticid, sector=sectors[0])
        data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True, save_postcard=False)
        q = data.quality == 0
        # create raw flux light curve
        raw_lc = lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], label='raw', time_format='btjd').remove_nans().normalize()
        corr_lc = lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], label='corr', time_format='btjd').remove_nans().normalize()
        pca_lc = lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], label='pca', time_format='btjd').remove_nans().normalize()
        psf_lc = lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], label='psf', time_format='btjd').remove_nans().normalize()
        #track breakpoints between sectors
        self.breakpoints = [raw_lc.time[-1]]
        # iterate through extra sectors and append the light curves
        if len(sectors) > 1:
            for s in sectors[1:]:
                star = eleanor.Source(tic=ticid, sector=s, tc=True)
                data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True, save_postcard=False)
                q = data.quality == 0

                raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], time_format='btjd').remove_nans().normalize())
                corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], time_format='btjd').remove_nans().normalize())
                pca_lc = pca_lc.append(lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], time_format='btjd').remove_nans().normalize())
                psf_lc = psf_lc.append(lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], time_format='btjd').remove_nans().normalize())

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
        _, outliers = lc.remove_outliers(sigma=4.5, return_mask=True)
        mask = momdump | outliers | burnin
        lc.time = lc.time[~mask]
        lc.flux = lc.flux[~mask]
        lc.flux_err = lc.flux_err[~mask]
        lc.flux = lc.flux - 1
        lc.flux = lc.flux - scipy.ndimage.filters.median_filter(lc.flux, 150)#changed Gaussian to median

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
            print('flux:',np.mean(flux))
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

        # mask first 12h after momentum dump
        momdump = (time > 1339) * (time < 1341)
        momdump2 = (time > 1541) * (time < 1545)
        momdump3 = (time > 1555.5) * (time < 1558)
        momdump4 = (time > 1568.5) * (time < 1571.3)
        momdump5 = (time > 1581) * (time < 1583.5)
        momdump6 = (time > 1535) * (time < 1535.5)
        # also the burn in
        burnin = np.zeros_like(time, dtype=bool)
        burnin[:20] = True

        #for special cases: Sam added
        if ticid == 'TIC 146660530':
            burnin[-1100:] = True
        elif ticid == 'TIC 266728763':
            burnin[-600:] = True
        elif ticid == 'TIC 452039751':
            burnin[-600:] = True


        # also 6 sigma outliers
        _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        mask = momdump | momdump2 | momdump3 | momdump4 | momdump5 | momdump6 | burnin | outliers
        time = time[~mask]
        flux = flux[~mask]
        flux_err = flux_err[~mask]


        # store masked values
        self.lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

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
        periods = np.linspace(2,20,400)
        #print(periods)
        results = model.power(periods, 0.16)
        #results = model.autopower(0.16)
        period = results.period[np.argmax(results.power)]
        t0 = results.transit_time[np.argmax(results.power)]
        dur = results.duration[np.argmax(results.power)]
        print('period:', period, 't0:', t0, 'duration:', dur)

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
        np.savetxt(outdir+'/'+str(ticid)+'.dat.ts', np.transpose([time,flux]), fmt='%.8f', delimiter='     ')
        np.savetxt(outdir+'/'+str(ticid)+'.dat.ts.fft', np.transpose([freq, power]), fmt='%.8f', delimiter='     ')



    def validate_transit(self, ticid=None, lc=None, rprs=0.02):
        """Take a closer look at potential transit signals."""
        from .utils import create_starry_model

        if ticid is not None:
            lc = self.from_eleanor(ticid)[1]
            #lc = self._clean_data(lc)
        elif lc is None:
            lc = self.lc

        lc = self._clean_data(lc)
        time, flux, flux_err = lc.time, lc.flux, lc.flux_err

        # mask first 12h after momentum dump
        momdump = (time > 1339) * (time < 1341)
        momdump2 = (time > 1541) * (time < 1545)
        momdump3 = (time > 1555.5) * (time < 1558)
        momdump4 = (time > 1568.5) * (time < 1571.3)
        momdump5 = (time > 1581) * (time < 1583.5)
        momdump6 = (time > 1535) * (time < 1535.5)
        # also the burn in
        burnin = np.zeros_like(time, dtype=bool)
        #burnin[:120] = True
        # also 6 sigma outliers
        _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        mask = momdump | momdump2 | momdump3 | momdump4 | momdump5 | momdump6 | burnin | outliers
        time = time[~mask]
        flux = flux[~mask]
        flux_err = flux_err[~mask]

        # store masked values
        self.lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        #lc.flux = lc.flux / np.median(lc.flux)

        model = BoxLeastSquares(lc.time, lc.flux)
        periods = np.linspace(2,20,400)
        #print(periods)
        results = model.power(periods, 0.16)
        #results = model.autopower(0.16)
        period = results.period[np.argmax(results.power)]
        print('period:', period)
        t0 = results.transit_time[np.argmax(results.power)]

        if rprs is None:
            depth = results.depth[np.argmax(results.power)]
            rprs = depth ** 2

        # create the model
        model_flux = create_starry_model(lc.time, period=period, t0=t0, rprs=rprs) - 1
        #print('model_flux:',model_flux)
        model_lc = lk.LightCurve(time=lc.time, flux=model_flux)
        #print(model_lc.time, model_lc.flux)

        fig, ax = plt.subplots(3, 1, figsize=(12,14))
        '''
        Plot folded transit
        -------------------
        '''
        lc.scatter(ax=ax[0], c='k', label='Corrected Flux')
        model_lc.plot(ax=ax[0], c='r', lw=2, label='Transit Model')
        #plt.plot(model_lc.time, model_lc.flux, c='r', lw=2, label='Transit Model')
        #ax[0].set_ylim([-.002, .002])
        #ax[0].set_xlim([lc.time[0], lc.time[-1]])

        '''
        Plot unfolded transit
        ---------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[1], c='k', label=f'folded at {period:.3f} days')
        lc.fold(period, t0).bin(binsize=7).plot(ax=ax[1], c='b', label='binned', lw=2)
        model_lc.fold(period, t0).plot(ax=ax[1], c='r', lw=2, label="transit Model")
        ax[1].set_xlim([-0.5, .5])
        #ax[1].set_ylim([-.002, .002])

        '''
        Zoom unfolded transit
        ---------------------
        '''
        lc.fold(period, t0).scatter(ax=ax[2], c='k', label=f'folded at {period:.3f} days')
        lc.fold(period, t0).bin(binsize=7).plot(ax=ax[2], c='b', label='binned', lw=2)
        model_lc.fold(period, t0).plot(ax=ax[2], c='r', lw=2, label="transit Model")
        ax[2].set_xlim([-0.1, .1])
        #ax[2].set_ylim([-.002, .002])

        ax[0].set_title(f'{ticid}', fontsize=14)

        plt.show()

    def validate_ktransit(self, ticid=None, lc=None, rprs=0.02):
        import ktransit
        from ktransit import FitTransit
        fitT = FitTransit()

        if ticid is not None:
            lc = self.from_eleanor(ticid)[1]
            #lc = self._clean_data(lc)
        elif lc is None:
            lc = self.lc

        #lc = self._clean_data(lc)
        time, flux, flux_err = lc.time, lc.flux, lc.flux_err

        ## # mask first 12h after momentum dump
        ## momdump = (time > 1339) * (time < 1341)
        ## momdump2 = (time > 1541) * (time < 1545)
        ## momdump3 = (time > 1555.5) * (time < 1558)
        ## momdump4 = (time > 1568.5) * (time < 1571.3)
        ## momdump5 = (time > 1581) * (time < 1583.5)
        ## momdump6 = (time > 1535) * (time < 1538)
        ## # also the burn in
        ## burnin = np.zeros_like(time, dtype=bool)
        ## burnin[:12] = True
        ## # also 6 sigma outliers
        ## _, outliers = lc.remove_outliers(sigma=6, return_mask=True)
        ## mask = momdump | momdump2 | momdump3 | momdump4 | momdump5 | momdump6 | burnin | outliers
        ## time = time[~mask]
        ## flux = flux[~mask]
        ## flux_err = flux_err[~mask]

        # store masked values
        self.lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        #lc.flux = lc.flux / np.median(lc.flux)
        model = BoxLeastSquares(lc.time, lc.flux)
        #results = model.autopower(0.16)
        periods = np.linspace(2,20,400)
        results = model.power(periods, 0.16)
        period = results.period[np.argmax(results.power)]
        t0 = results.transit_time[np.argmax(results.power)]
        if rprs is None:
            depth = results.depth[np.argmax(results.power)]
            rprs = depth ** 2


        print('params:period, t0, rprs',period,t0, rprs)
        fitT.add_guess_star(rho=0.008, zpt=0, ld1=0.6505,ld2=0.1041) #come up with better way to estimate this using AS
        fitT.add_guess_planet(T0=t0, period=period, impact=0.3, rprs=rprs)

        ferr=np.ones_like(lc.time) * 0.00001
        fitT.add_data(time=lc.time,flux=lc.flux,ferr=ferr)#*1e-3)

        vary_star = ['zpt']      # free stellar parameters
        vary_planet = ([ 'period',# 'impact',       # free planetary parameters
             #'esinw', 'ecosw',
            'T0', 'rprs']) #'impact',               # free planet parameters are the same for every planet you model

        fitT.free_parameters(vary_star, vary_planet)
        fitT.do_fit()                   # run the fitting

        fitT.print_results()            # print some results
        res=fitT.fitresultplanets
        res2=fitT.fitresultstellar

        fig = ktransit.plot_results(lc.time,lc.flux,fitT.transitmodel)

        fig.show()

    def plot_gaia_overlay(self, ticid=None, tpf=None):
        """Check if the source is contaminated."""
        from .utils import add_gaia_figure_elements

        if ticid is None:
            ticid = self.ticid

        if tpf is None:
            try:
                tpf = lk.search_tesscut(ticid)[0].download(cutout_size=9)
            except:
                star = eleanor.Source(tic=ticid)#, sector=sectors[0])#, tc=True)
                data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True, save_postcard=False)
                tpf = data.get_tpf_from_postcard(pos=data.post_obj.center_xy, postcard=data.post_obj, height=15, width=15, bkg_size=31, save_postcard=False, source=star)

        fig = tpf.plot()
        fig = add_gaia_figure_elements(tpf, fig)

        return fig

class CPM(object):
    """
    """
    def __init__(self, fits_file, remove_bad="True"):
        self.file_name = fits_file.split("/")[-1]  # Should I really keep this here?
        with fits.open(fits_file, mode="readonly") as hdulist:
            self.time = hdulist[1].data["TIME"]
            self.im_fluxes = hdulist[1].data["FLUX"]  # Shape is (1282, 64, 64)
            self.im_errors = hdulist[1].data["FLUX_ERR"]  # Shape is (1282, 64, 64)
            self.quality = hdulist[1].data["QUALITY"]

        # If remove_bad is set to True, we'll remove the values with a nonzero entry in the quality array
        if remove_bad == True:
            print("Removing bad values by using the TESS provided \"QUALITY\" array")
            b = (self.quality == 0)  # The zero value entries for quality array are the "good" values
            self.time = self.time[b]
            self.im_fluxes = self.im_fluxes[b]
            self.im_errors = self.im_errors[b]

        # Calculate the vandermode matrix to add polynomial components to model
        self.scaled_centered_time = ((self.time - (self.time.max() + self.time.min())/2)
                                    / (self.time.max() - self.time.min()))
        self.v_matrix = np.vander(self.scaled_centered_time, N=4, increasing=True)

        self.target_row = None
        self.target_col = None
        self.target_fluxes = None
        self.target_errors = None
        self.target_median = None
        self.rescaled_target_fluxes = None
        self.rescaled_target_errors = None
        self.target_pixel_mask = None

        self.excluded_pixels_mask = None

        # We're going to precompute the pixel lightcurve medians since it's used to set the predictor pixels
        # but never has to be recomputed
        self.pixel_medians = np.median(self.im_fluxes, axis=0)
        self.flattened_pixel_medians = self.pixel_medians.reshape(self.im_fluxes[0].shape[0]**2)

        # We'll precompute the rescaled values for the fluxes (F* = F/M - 1)
        self.rescaled_im_fluxes = (self.im_fluxes/self.pixel_medians) - 1

        self.method_predictor_pixels = None
        self.num_predictor_pixels = None
        self.predictor_pixels_locations = None
        self.predictor_pixels_mask = None
        self.predictor_pixels_fluxes = None
        self.rescaled_predictor_pixels_fluxes = None

        self.fit = None
        self.regularization = None
        self.lsq_params = None
        self.cpm_params = None
        self.poly_params = None
        self.cpm_prediction = None
        self.poly_prediction = None
        self.prediction = None
        self.im_predicted_fluxes = None
        self.im_diff = None

        self.is_target_set = False
        self.is_exclusion_set = False
        self.are_predictors_set = False
        self.trained = False

    def set_target(self, target_row, target_col):
        self.target_row = target_row
        self.target_col = target_col
        self.target_fluxes = self.im_fluxes[:, target_row, target_col]  # target pixel lightcurve
        self.target_errors = self.im_errors[:, target_row, target_col]  # target pixel errors
        self.target_median = np.median(self.target_fluxes)
        self.rescaled_target_fluxes = self.rescaled_im_fluxes[:, target_row, target_col]
        self.rescaled_target_errors = self.target_errors / self.target_median

        target_pixel = np.zeros(self.im_fluxes[0].shape)
        target_pixel[target_row, target_col] = 1
        self.target_pixel_mask = np.ma.masked_where(target_pixel == 0, target_pixel)  # mask to see target

        self.is_target_set = True

    def set_exclusion(self, exclusion, method="cross"):
        if self.is_target_set == False:
            print("Please set the target pixel to predict using the set_target() method.")
            return

        r = self.target_row  # just to reduce verbosity for this function
        c = self.target_col
        exc = exclusion
        im_side_length = self.im_fluxes.shape[1]  # for convenience

        excluded_pixels = np.zeros(self.im_fluxes[0].shape)
        if method == "cross":
            excluded_pixels[max(0,r-exc) : min(r+exc+1, im_side_length), :] = 1
            excluded_pixels[:, max(0,c-exc) : min(c+exc+1, im_side_length)] = 1

        if method == "row_exclude":
            excluded_pixels[max(0,r-exc) : min(r+exc+1, im_side_length), :] = 1

        if method == "col_exclude":
            excluded_pixels[:, max(0,c-exc) : min(c+exc+1, im_side_length)] = 1

        if method == "closest":
            excluded_pixels[max(0,r-exc) : min(r+exc+1, im_side_length),
                            max(0,c-exc) : min(c+exc+1, im_side_length)] = 1

        self.excluded_pixels_mask = np.ma.masked_where(excluded_pixels == 0, excluded_pixels)  # excluded pixel is "valid" and therefore False
        self.is_exclusion_set = True

    def set_predictor_pixels(self, num_predictor_pixels, method="similar_brightness", seed=None):
        if seed != None:
            np.random.seed(seed=seed)

        if (self.is_target_set == False) or (self.is_exclusion_set == False):
            print("Please set the target pixel and exclusion.")
            return

        self.method_predictor_pixels = method
        self.num_predictor_pixels = num_predictor_pixels
        im_side_length = self.im_fluxes.shape[1]  # for convenience (I need column size to make this work)

        # I'm going to do this in 1D by assinging individual pixels a single index instead of two.
        coordinate_idx = np.arange(im_side_length**2)
        possible_idx = coordinate_idx[self.excluded_pixels_mask.mask.ravel()]

        if method == "random":
            chosen_idx = np.random.choice(possible_idx, size=num_predictor_pixels, replace=False)

        # Since it turns out that calculating the median of the pixels across time is somewhat expensive,
        # we're going to optimize the following by doing the median calculation at the time of instantiating
        # the CPM instead of calculating it each time the function is used.
#         if method == "similar_brightness":
#             target_median = np.median(self.target_fluxes)  # Median value of target lightcurve
#             pixel_medians = np.median(self.im_fluxes, axis=0)  # 64x64 matrix of median values of lightcurves

#             flattened_pixel_medians = pixel_medians.reshape(im_side_length**2)  # 4096 length array
#             possible_pixel_medians = flattened_pixel_medians[self.excluded_pixels_mask.mask.ravel()]

#             diff = (np.abs(possible_pixel_medians - target_median))
#             chosen_idx = possible_idx[np.argsort(diff)[0:self.num_predictor_pixels]]

        # Optimized version of above
        if method == "similar_brightness":
#             target_median = np.median(self.target_fluxes)

            possible_pixel_medians = self.flattened_pixel_medians[self.excluded_pixels_mask.mask.ravel()]
#             diff = (np.abs(possible_pixel_medians - target_median))
            diff = (np.abs(possible_pixel_medians - self.target_median))

            chosen_idx = possible_idx[np.argsort(diff)[0:self.num_predictor_pixels]]

        self.predictor_pixels_locations = np.array([[idx // im_side_length, idx % im_side_length]
                                                   for idx in chosen_idx])
        loc = self.predictor_pixels_locations.T
        predictor_pixels = np.zeros((self.im_fluxes[0].shape))
        predictor_pixels[loc[0], loc[1]] = 1

        self.predictor_pixels_fluxes = self.im_fluxes[:, loc[0], loc[1]]  # shape is (1282, num_predictors)
        self.rescaled_predictor_pixels_fluxes = self.rescaled_im_fluxes[:, loc[0], loc[1]]
        self.predictor_pixels_mask = np.ma.masked_where(predictor_pixels == 0, predictor_pixels)

        self.are_predictors_set = True

    def train(self, reg):
        if ((self.is_target_set  == False) or (self.is_exclusion_set == False)
           or self.are_predictors_set == False):
            print("You missed a step.")

        def objective(coeff, reg):
            model = np.dot(coeff, self.predictor_pixels_fluxes.T)
            chi2 = ((self.target_fluxes - model)/(self.target_errors))**2
            return np.sum(chi2) + reg*np.sum(coeff**2)

        init_coeff = np.zeros(self.num_predictor_pixels)
        self.fit = minimize(objective, init_coeff, args=(reg), tol=0.5)
        self.prediction = np.dot(self.fit.x, self.predictor_pixels_fluxes.T)
        print(self.fit.success)
        print(self.fit.message)

        self.trained = True

    def lsq(self, reg, rescale=True, polynomials=False):
        if ((self.is_target_set  == False) or (self.is_exclusion_set == False)
           or self.are_predictors_set == False):
            print("You missed a step.")

        self.regularization = reg
        num_components = self.num_predictor_pixels

        if (rescale == False):
            print("Calculating parameters using unscaled values.")
            y = self.target_fluxes
            m = self.predictor_pixels_fluxes  # (num of measurements(1282) , num of predictors (128))

        elif (rescale == True):
            y = self.rescaled_target_fluxes
            m = self.rescaled_predictor_pixels_fluxes

        if (polynomials == True):
            m = np.hstack((m, self.v_matrix))
            num_components = num_components + self.v_matrix.shape[1]

        l = reg*np.identity(num_components)
        a = np.dot(m.T, m) + l
        b = np.dot(m.T, y)

        self.lsq_params = np.linalg.solve(a, b)
        self.cpm_params = self.lsq_params[:self.num_predictor_pixels]
        self.poly_params = self.lsq_params[self.num_predictor_pixels:]
        self.cpm_prediction = np.dot(m[:, :self.num_predictor_pixels], self.cpm_params)
        self.poly_prediction = np.dot(m[:, self.num_predictor_pixels:], self.poly_params)
        self.lsq_prediction = np.dot(m, self.lsq_params)

        if (rescale == True):
            self.lsq_prediction = np.median(self.target_fluxes)*(self.lsq_prediction + 1)
            if (polynomials == True):
                self.cpm_prediction = np.median(self.target_fluxes)*(self.cpm_prediction + 1)
                self.poly_prediction = np.median(self.target_fluxes)*(self.poly_prediction + 1)

        self.trained = True

    def get_contributing_pixels(self, number):
        """Return the n-most contributing pixels' locations and a mask to see them"""
        if self.trained == False:
            print("You need to train the model first.")


        if self.fit == None:
            idx = np.argsort(np.abs(self.cpm_params))[:-(number+1):-1]
        else:
            idx = np.argsort(np.abs(self.fit.x))[:-(number+1):-1]

        top_n_loc = self.predictor_pixels_locations[idx]
        loc = top_n_loc.T
        top_n = np.zeros(self.im_fluxes[0].shape)
        top_n[loc[0], loc[1]] = 1

        top_n_mask = np.ma.masked_where(top_n == 0, top_n)

        return (top_n_loc, top_n_mask)

    def entire_image(self, reg):
        self.reg = reg
        self.im_predicted_fluxes = np.empty(self.im_fluxes.shape)
        num_col = self.im_fluxes[0].shape[1]
        idx = np.arange(num_col**2)
        rows = idx // num_col
        cols = idx % num_col
        for (row, col) in zip(rows, cols):
#         for (row, col) in zip(rows[:10], cols[:10]):
            self.set_target(row, col)
            self.set_exclusion(4, method="cross")
            self.set_predictor_pixels(128, method="similar_brightness")
            self.lsq(reg, rescale=True, polynomials=True)
            self.im_predicted_fluxes[:, row, col] = self.cpm_prediction
        self.im_diff = self.im_fluxes - self.im_predicted_fluxes
