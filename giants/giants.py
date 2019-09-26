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

class Giant(object):

    def __init__(self):
        self.cvz = self.get_cvz_targets()
        self.brightcvz = self.cvz.GAIAmag < 6.5

        print(f'Using the brightest {len(self.cvz[self.brightcvz])} targets.')

    def get_cvz_targets(self):
        try:
            # full list (too big to upload to github)
            path = os.path.abspath(os.path.join(PACKAGEDIR, 'data', 'TICgiants_bright.csv'))
        except:
            # shorter list
            path = os.path.abspath(os.path.join(PACKAGEDIR, 'data', 'TICgiants_CVZ.csv'))
        return pd.read_csv(path, skiprows=4)

    def from_lightkurve(self, ind=0, ticid=None, pld=True):
        '''
        Returns
        -------
        LightCurveCollection
        '''
        if ticid == None:
            i = ind
            ticid = self.cvz[self.brightcvz].ID.values[i]
        print(ticid)
        tpfc = self.get_data(ticid=ticid)
        rlc = self.photometry(tpfc[0], pld=False).normalize()
        self.breakpoints = [rlc.time[-1]]
        for t in tpfc[1:]:
            single_rlc = self.photometry(t, pld=False).normalize()
            rlc = rlc.append(single_rlc)
            self.breakpoints.append(single_rlc.time[-1])
        rlc.label = 'Raw {ticid}'
        if pld:
            clc = self.photometry(tpfc[0], pld=True).normalize()
            for t in tpfc[1:]:
                single_clc = self.photometry(t, pld=True).normalize()
                clc = clc.append(single_clc)
            clc.label = 'PLD {ticid}'
            rlc = rlc.remove_nans()
            clc = clc.remove_nans()
            return lk.LightCurveCollection([rlc, clc])
        else:
            rlc = rlc.remove_nans()
            return lk.LightCurveCollection([rlc])

    def get_data(self, ticid):
        # search for targets
        sr = lk.search_tesscut('tic{}'.format(ticid))
        # download a tpf collection
        tpfc = sr.download_all(cutout_size=5)
        return tpfc

    def from_eleanor(self, ticid):
        sr = lk.search_tesscut(ticid)
        sectors = []
        for desc in sr.table['description']:
            sectors.append(int(re.search(r'\d+', str(desc)).group()))

        print(f'Creating light curve for target {ticid} for sectors {sectors}.')

        star = eleanor.Source(tic=ticid, sector=sectors[0], tc=True)
        data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True)
        q = data.quality == 0

        # create raw flux light curve
        raw_lc = lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], label='raw').normalize()
        corr_lc = lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], label='corr').normalize()
        pca_lc = lk.LightCurve(time=data.time[q], flux=data.pca_flux[q], label='pca').normalize()
        psf_lc = lk.LightCurve(time=data.time[q], flux=data.psf_flux[q], label='psf').normalize()

        self.breakpoints = [raw_lc.time[-1]]

        if len(sectors) > 1:
            for s in sectors[1:]:
                star = eleanor.Source(tic=ticid, sector=s, tc=True)
                data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True, try_load=True)
                q = data.quality == 0

                raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q]).normalize())
                corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q]).normalize())
                pca_lc = pca_lc.append(lk.LightCurve(time=data.time[q], flux=data.pca_flux[q]).normalize())
                psf_lc = psf_lc.append(lk.LightCurve(time=data.time[q], flux=data.psf_flux[q]).normalize())

                self.breakpoints.append(raw_lc.time[-1])

        return lk.LightCurveCollection([raw_lc, corr_lc, pca_lc, psf_lc])

    def photometry(self, tpf, pld=True):
        if pld:
            pld = tpf.to_corrector('pld')
            lc = pld.correct(aperture_mask='threshold', pld_aperture_mask='all', use_gp=False)
        else:
            lc = tpf.to_lightcurve(aperture_mask='threshold')
        return lc

    def plot(self, ticid, use='eleanor'):

        plt.clf()

        '''
        Plot Light Curve
        ----------------
        '''
        plt.subplot2grid((4,4),(0,0),colspan=2)

        if use == 'lightkurve':
            lcc = self.from_lightkurve(ticid)
            q = lcc[0].quality == 0

            plt.plot(lcc[0].time[q], lcc[0].flux[q], 'k', label="Raw")
            if len(lcc) > 1:
                plt.plot(lcc[1].time[q], lcc[1].flux[q], 'r', label="Corr")
            for val in self.breakpoints:
                plt.axvline(val, c='b', linestyle='dashed')
            lc = lcc[-1]
            time = lc.time[q]
            flux = lc.flux[q] - 1
            flux = flux - scipy.ndimage.filters.gaussian_filter(flux, 100)
            flux_err = lc.flux_err[q]

        else:
            lcc = self.from_eleanor(ticid)
            for lc,label in zip(lcc, ['raw', 'corr', 'pca', 'psf']):
                plt.plot(lc.time, lc.flux, label=label)
            for val in self.breakpoints:
                plt.axvline(val, c='b', linestyle='dashed')
            plt.legend(loc=0)

            lc = lcc[1] # using corr_lc
            time = lc.time
            flux = lc.flux - 1
            flux = flux - scipy.ndimage.filters.gaussian_filter(flux, 100)
            flux_err = np.ones_like(flux) * 1e-5


        #mask first 12h after momentum dump
        momdump = (time > 1339) * (time < 1341)
        time = time[~momdump]
        flux = flux[~momdump]

        plt.legend()
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')


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

        if use == 'lightkurve':
            power = lc.to_periodogram().power
            freq = np.linspace(1./15, 1./.01, len(power))
        else:
            freq = np.linspace(1./15, 1./.01, 100000)
            power = ass.LombScargle(time, flux, flux_err).power(freq)
        ps = 1./freq

        '''
        Plot Periodogram
        ----------------
        '''
        plt.subplot2grid((4,4),(0,2),colspan=2,rowspan=4)
        plt.loglog(freq/24/3600 * 1e6, power)
        plt.loglog(freq/24/3600 * 1e6, scipy.ndimage.filters.gaussian_filter(power, 500), color='r', alpha=0.8, lw=2.5)
        plt.axvline(283,-1,1, ls='--', color='k')
        plt.xlabel("Frequency [uHz]")
        plt.ylabel("Power")
        plt.xlim(10, 400)

        '''
        Plot BLS
        --------
        '''
        plt.subplot2grid((4,4),(2,0),colspan=2)

        model = BoxLeastSquares(time, flux)
        results = model.autopower(0.16)
        period = results.period[np.argmax(results.power)]

        plt.plot(results.period, results.power, "k", lw=0.5)
        plt.xlim(results.period.min(), results.period.max())
        plt.xlabel("period [days]")
        plt.ylabel("log likelihood")

        # Highlight the harmonics of the peak period
        plt.axvline(period, alpha=0.4, lw=4)
        for n in range(2, 10):
            plt.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
            plt.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")

        foldedtimes = time % (period)
        foldtimesort = np.argsort(foldedtimes)
        foldfluxes = flux[foldtimesort]
        plt.subplot2grid((4,4),(3,0),colspan=2)
        plt.scatter(foldedtimes, flux, s=2)
        plt.plot(np.sort(foldedtimes), scipy.ndimage.filters.median_filter(foldfluxes,40), lw=2, color='r')
        plt.xlabel('Time (d)')
        plt.ylabel('Flux')
        plt.xlim(0, period)

        fig = plt.gcf()
        fig.set_size_inches(12,10)
        plt.show()
