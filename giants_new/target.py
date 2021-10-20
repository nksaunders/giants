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


class Target(object):
    """
    """
    def __init__(self):
        pass

    def _find_sectors(self, ticid):
        """Hidden function to read sectors from a search result."""
        sectors = []

        search_result = lk.search_tesscut(f'TIC {ticid}')
        for sector in search_result.table['description']:
            sectors.append(int(re.search(r'\d+', sector).group()))
        return sectors

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

    def from_lightkurve(self, sectors=None, method='pca', flatten=True, **kwargs):
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
        if method=='pld':
            lc = self.pld(self.tpf)
        else:
            if flatten:
                lc = self.simple_pca(self.tpf).flatten(201)
            else:
                lc = self.simple_pca(self.tpf)
            raw_lc = self.tpf.to_lightcurve(aperture_mask='threshold')

        # store as LCC for plotting later
        self.lcc = lk.LightCurveCollection([lc])
        self.breakpoints = [lc.time[-1].value]
        for tpf in tpfc[1:]:
            if method=='pld':
                new_lc = self.pld(tpf)
            else:
                if flatten:
                    new_lc = self.simple_pca(tpf).flatten(201)
                else:
                    new_lc = self.simple_pca(tpf)
                new_raw_lc = tpf.to_lightcurve(aperture_mask='threshold')
            self.breakpoints.append(new_lc.time[-1].value)
            self.lcc.append(new_lc)
            lc = lc.append(new_lc)
            raw_lc = raw_lc.append(new_raw_lc)

        self.lc = lc
        self.raw_lc = raw_lc

        return lc

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
        self.breakpoints = [raw_lc.time[-1].value]
        # iterate through extra sectors and append the light curves
        if len(sectors) > 1:
            for s in sectors[1:]:
                try: # some sectors fail randomly
                    star = eleanor.Source(tic=self.ticid, sector=int(s), tc=True)
                    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False, try_load=True)
                    q = data.quality == 0

                    raw_lc = raw_lc.append(lk.LightCurve(time=data.time[q], flux=data.raw_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())
                    corr_lc = corr_lc.append(lk.LightCurve(time=data.time[q], flux=data.corr_flux[q], flux_err=data.flux_err[q], time_format='btjd').remove_nans().normalize())

                    self.breakpoints.append(raw_lc.time[-1].value)
                except:
                    continue
        # store in a LightCurveCollection object and return
        return lk.LightCurveCollection([raw_lc, corr_lc])

    def from_local_data(self, local_data_path):
        """
        Download data from local data cube.
        Data cubes should be stored in the format 's0001-1-1.fits'
        """

        sectors = self._find_sectors(self.ticid)
        if not self.silent:
            print(f'Creating light curve for target {self.ticid} for sectors {sectors}.')

        my_cutter = CutoutFactory()
        local_data_path = '/data/users/nsaunders/cubes' # !! HACK
        for obs in available_obs:
            cube_file = os.path.join(local_data_path,
                                     's{obs[0]:04d}-{obs[1]}-{obs[2]}.fits')

            cutout_file = my_cutter.cube_cut(cube_file, f'{self.RA}, {self.dec}', 5, verbose=True)


    def fetch_and_clean_data(self, lc_source='lightkurve', method='pca', flatten=True, sectors=None, gauss_filter_lc=True, **kwargs):
        """
        Query and download data, remove background signal and outliers. The light curve is stored as a
        object variable `Target.lc`.

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
            time = lc.time.value
            flux = lc.flux
            flux_err = np.ones_like(flux) * 1e-5
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

        elif lc_source == 'lightkurve':
            lc = self.from_lightkurve(sectors=sectors, method=method, flatten=flatten)

        lc = self._clean_data(lc, gauss_filter_lc=gauss_filter_lc)

    def _clean_data(self, lc, gauss_filter_lc=True):
        """Hidden function to remove common sources of noise and outliers."""
        # mask first 12h after momentum dump
        momdump = (lc.time.value > 1339) * (lc.time.value < 1341)

        # also the burn in
        burnin = np.zeros_like(lc.time.value, dtype=bool)
        burnin[:30] = True
        with open('data/downlinks.txt') as f:
            downlinks = [float(val.strip()) for val in f.readlines()]
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
        lc.flux = lc.flux - 1 * lc.flux.unit
        if gauss_filter_lc:
            lc.flux = lc.flux - scipy.ndimage.filters.gaussian_filter(lc.flux, 100) *lc.flux.unit # <2-day (5muHz) filter

        # store cleaned lc
        self.lc = lc
        self.mask = mask
        return lc