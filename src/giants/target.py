import os
import re
import numpy as np
import pandas as pd
import scipy
import lightkurve as lk
import warnings
from tess_stars2px import tess_stars2px_function_entry
from astrocut import CutoutFactory
from astroquery.mast import Catalogs

from . import PACKAGEDIR
from .plotting import plot_summary

# suppress verbose astropy warnings and future warnings
warnings.filterwarnings("ignore", module="astropy")
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ['Target']

class Target(object):
    """
    A class to hold a TESS target and its data.

    Parameters
    ----------
    ticid : int
        TIC ID of the target
    silent : bool
        suppress print statements
    """

    def __init__(self, ticid, silent=False):

        # parse TIC ID
        self.ticid = ticid
        if isinstance(self.ticid, str):
            self.ticid = int(re.search(r'\d+', str(self.ticid)).group())

        self.PACKAGEDIR = PACKAGEDIR
        self.has_target_info = False
        self.silent = silent

        self.search_result = lk.search_tesscut(f'TIC {ticid}')
        self.get_target_info(self.ticid)
        self.available_sectors = self.check_available_sectors()

    def __repr__(self):

        return f'giants.Target: TIC {self.ticid} (available sectors: {", ".join([str(s) for s in self.available_sectors])})'

    def get_target_info(self, ticid):
        """
        Get basic information about the target from the TIC catalog.

        Parameters
        ----------
        ticid : int
            TIC ID of the target
        """
        catalog_data = Catalogs.query_criteria(objectname=f'TIC {ticid}', catalog="Tic", radius=.0001, Bmag=[0,20])
        
        self.ra = catalog_data['ra'][0]
        self.dec = catalog_data['dec'][0]
        self.coords = f'({self.ra:.2f}, {self.dec:.2f})'
        self.rstar = catalog_data['rad'][0]
        self.teff = catalog_data['Teff'][0]

        self.has_target_info = True

    def check_available_sectors(self, ticid=None):
        """
        Helper function to check which sectors are available in the TESSCut search result.

        Parameters
        ----------
        ticid : int
            TIC ID of the target

        Returns
        -------
        available_sectors : list of ints
            list of available sectors
        """
        if ticid is not None:
            temp_search_result = lk.search_tesscut(f'TIC {ticid}')
        else:
            temp_search_result = self.search_result
        
        available_sectors = []
        for sector in temp_search_result.table['description']:
            available_sectors.append(int(re.search(r'\d+', sector).group()))

        return available_sectors
    
    def from_lightkurve(self, sectors=None, flatten=True, **kwargs):
        """
        Use `lightkurve.search_tesscut` to query and download TESSCut 11x11 cutout for target.
        This function creates a background model and subtracts it off using `lightkurve.RegressionCorrector`.

        Parameters
        ----------
        sectors : int, list of ints
            desired sector number or list of sector numbers
        flatten : bool
            optionally flatten the light curve
        **kwargs : dict
            additional keyword arguments to pass to `lightkurve.search_tesscut`

        Returns
        -------
        lc : `lightkurve.LightCurve` object
            background-corrected flux time series
        """

        # apply sector mask
        if sectors is not None:
            # make sure sectors is a list
            if isinstance(sectors, int):
                sectors = [sectors]

            search_result_mask = []
            for sector in self.available_sectors:
                search_result_mask.append(sector in sectors)

            masked_search_result = self.search_result[search_result_mask]
        else:
            masked_search_result = self.search_result

        # download data
        tpfc = lk.TargetPixelFileCollection([])
        for search_row in masked_search_result:
            try:
                tpfc.append(search_row.download(cutout_size=11))
            except:
                continue

        # apply the pca background correction
        self.tpf = tpfc[0]
        lc = self.apply_pca_corrector(self.tpf)
        raw_lc = self.tpf.to_lightcurve(aperture_mask='threshold')

        self.lcc = lk.LightCurveCollection([lc])
        self.breakpoints = [lc.time[-1].value]

        for tpf in tpfc[1:]:
            new_lc = self.apply_pca_corrector(tpf)
            new_raw_lc = tpf.to_lightcurve(aperture_mask='threshold')

            # flatten lc
            if flatten:
                lc = lc.flatten()

            # stitch together
            lc = lc.append(new_lc)
            raw_lc = raw_lc.append(new_raw_lc)

        self.lc = lc
        self.raw_lc = raw_lc

        return lc
    
    def from_local_data(self, local_data_path, sectors=None, flatten=False):
        """
        Retrieve data from local data cube.
        Data cubes should be stored in the format 's0001-1-1.fits'

        Parameters
        ----------
        local_data_path : str
            path to local data cube
        sectors : int, list of ints
            desired sector number or list of sector numbers
        flatten : bool
            optionally flatten the light curve

        Returns
        -------
        lc : `lightkurve.LightCurve` object
            background-corrected flux time series
        """

        self.get_target_info(self.ticid)

        obs = self.fetch_obs(self.ra, self.dec)
        sectors = obs[0]
        if not self.silent:
            print(f'Creating light curve for target {self.ticid} for sectors {sectors}.')

        my_cutter = CutoutFactory()
        local_data_path = '/data/users/nsaunders/cubes' # !! HACK
        available_obs = np.array(obs).T.reshape(len(obs[0]), len(obs))
        tpfs = []
        for obs_ in available_obs:
            try:
                cube_file = os.path.join(local_data_path,
                                         f's{obs_[0]:04d}-{obs_[1]}-{obs_[2]}.fits')

                cutout_file = my_cutter.cube_cut(cube_file, f'{self.ra} {self.dec}', 11, verbose=False)
                tpfs.append(lk.read(cutout_file))
                os.remove(cutout_file)
            except:
                continue

        tpfc = lk.TargetPixelFileCollection(tpfs)

        self.tpf = tpfc[0]
        if flatten:
            lc = self.apply_pca_corrector(self.tpf).flatten(1001)
        else:
            lc = self.apply_pca_corrector(self.tpf)

        # store as LCC for plotting later
        self.lcc = lk.LightCurveCollection([lc])
        self.breakpoints = [lc.time[-1]]
        for tpf in tpfc[1:]:
            if flatten:
                new_lc = self.apply_pca_corrector(tpf).flatten(1001)
            else:
                new_lc = self.apply_pca_corrector(tpf)
            self.breakpoints.append(new_lc.time[-1])
            self.lcc.append(new_lc)
            lc = lc.append(new_lc)

        self.lc = lc

        return lc
    
    def fetch_obs(self, ra, dec):
        """
        Query MAST for TESS observations of a given target.

        Parameters
        ----------
        ra : float
            right ascension of target
        dec : float
            declination of target

        Returns
        -------
        obs : list of lists
            list of observations, each of which is a list of [sector, camera, ccd]
        """

        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
                outColPix, outRowPix, scinfo = tess_stars2px_function_entry(
                        self.ticid, float(ra), float(dec))

        return outSec, outCam, outCcd
    
    def apply_pca_corrector(self, tpf, zero_point_background=False):
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
        zero_point_background : bool
            optionally normalize light curve to the 5th percentile of model flux. can be useful for targets 
            which raise warnings when flattened. default is False.

        Returns
        -------
        corrected_lc : `lightkurve.LightCurve` object
            background-corrected light curve
        """
        aper = tpf._parse_aperture_mask('threshold')
        raw_lc = tpf.to_lightcurve(aperture_mask=aper)
        mask = (raw_lc.flux_err > 0) | (~np.isnan(raw_lc.flux))
        self.raw_lc = raw_lc[mask]
        tpf = tpf[mask]


        regressors = tpf.flux[:, ~aper]

        dm = lk.DesignMatrix(regressors, name='regressors')

        dm = dm.pca(10)
        dm = dm.append_constant()

        corrector = lk.RegressionCorrector(self.raw_lc.normalize())
        corrected_lc_unnormalized = corrector.correct(dm)
        model = corrector.model_lc

        if zero_point_background:
            # Normalize to the 5th percentile of model flux
            model -= np.percentile(model.flux, 5)

        corrected_lc = lk.LightCurve(time=model.time, flux=self.raw_lc.normalize().flux.value-model.flux, flux_err=self.raw_lc.flux_err.value)

        return corrected_lc
    
    def fetch_and_clean_data(self, lc_source='lightkurve', flatten=True, sectors=None, gauss_filter_lc=True, **kwargs):
        """
        Query and download data, remove background signal and outliers. The light curve is stored as a
        object variable `Target.lc`.

        Parameters
        ----------
        lc_source : str, 'lightkurve' or 'local'
            pipeline used to access data
        flatten : bool
            optionally flatten the light curve
        sectors : int, list of ints
            desired sector number or list of sector numbers
        gauss_filer_lc : bool
            optionally apply Gaussian smoothing with a ~2 day filter (good for planets, bad for stars)
        **kwargs : dict
            additional keyword arguments to pass to `lightkurve.search_tesscut`

        Returns
        -------
        self : `Target` object
            returns self for chaining
        """
       
        if lc_source == 'lightkurve':
            lc = self.from_lightkurve(sectors=sectors, flatten=flatten)

        elif lc_source == 'local':
            lc = self.from_local_data('/data/users/nsaunders/cubes')

        lc = self._clean_data(lc, gauss_filter_lc=gauss_filter_lc)

        return self
    
    def _clean_data(self, lc, gauss_filter_lc=True):
        """Hidden function to remove common sources of noise and outliers."""
        # mask first 12h after momentum dump
        momdump = (lc.time.value > 1339) * (lc.time.value < 1341)

        # also the burn in
        burnin = np.zeros_like(lc.time.value, dtype=bool)
        burnin[:30] = True
        with open(os.path.join(self.PACKAGEDIR, 'data/downlinks.txt')) as f:
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
    
    def save_to_fits(self, outdir=None, lc_source='local'):
        """
        Pipeline to download and de-trend a target using the `lightkurve` implememtation.
        Downloads data, removes background, and saves as fits files. This function outputs:
         - {TICID}_s{SECTOR}_corr.fits : corrected light curve
         - {TICID}_s{SECTOR}_raw.fits : raw SAP flux light curve

        Parameters
        ----------
        outdir : str or path
            location of fits output
        lc_source : str, 'lightkurve' or 'local'
            pipeline used to access data

        Returns
        -------
        self : `Target` object
            returns self for chaining
        """
        self.silent = True

        if outdir is None:
            outdir = os.path.join(self.PACKAGEDIR, 'outputs')

        for s in self.available_sectors:
            self.fetch_and_clean_data(lc_source='lightkurve', sectors=s, gauss_filter_lc=False)

            fname_corr = f'{self.ticid}_s{s:02d}_corr.fits'
            fname_raw = f'{self.ticid}_s{s:02d}_raw.fits'

            path_corr = os.path.join(outdir, fname_corr)
            path_raw = os.path.join(outdir, fname_raw)

            self.lc.flux += 1.

            self.lc.to_fits(path=path_corr, overwrite=True)
            self.raw_lc.to_fits(path=path_raw, overwrite=True)

    def create_summary_plot(self, **kwargs):
        plot_summary(self, **kwargs)