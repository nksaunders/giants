import os
import re
import glob
import pathlib
import numpy as np
import pandas as pd
import scipy
import lightkurve as lk
import warnings
import astropy.units as u
from tess_stars2px import tess_stars2px_function_entry
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.config import set_temp_cache

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

    def __init__(self, ticid, target_info=None, silent=False):

        # parse TIC ID
        self.ticid = ticid
        if isinstance(self.ticid, str):
            self.ticid = int(re.search(r'\d+', str(self.ticid)).group())

        self.PACKAGEDIR = PACKAGEDIR
        self.has_target_info = False
        self.silent = silent

        if target_info is None:
            self.get_target_info(self.ticid)
        else:
            self.ra = target_info['ra']
            self.dec = target_info['dec']
            self.coords = SkyCoord(ra=self.ra, dec=self.dec, frame='icrs', unit=(u.deg, u.deg))
            self.rstar = target_info['rstar']
            self.mstar = target_info['mstar']
            self.teff = target_info['teff']
            self.logg = target_info['logg']
            self.vmag = target_info['vmag']
            self.has_target_info = True

        self.available_sectors, self.cameras, self.ccds = self.fetch_obs(self.ra, self.dec)
        # self.available_sectors = self.check_available_sectors()

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
        # with set_temp_cache(f'/home/nsaunders/mendel-nas1/temp_cache/tic{ticid}_cache', delete=True):
        catalog_data = Catalogs.query_criteria(objectname=f'TIC {ticid}', catalog="Tic", radius=.0001, Bmag=[0,20])
        
        self.ra = catalog_data['ra'][0]
        self.dec = catalog_data['dec'][0]
        self.coords = SkyCoord(ra=self.ra, dec=self.dec, frame='icrs', unit=(u.deg, u.deg))
        self.rstar = catalog_data['rad'][0]
        self.teff = catalog_data['Teff'][0]
        self.logg = catalog_data['logg'][0]
        self.vmag = catalog_data['Vmag'][0]

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
            self.search_result = lk.search_tesscut(f'TIC {ticid}')
        else:
            self.search_result = lk.search_tesscut(f'TIC {self.ticid}')
        
        available_sectors = []
        for sector in self.search_result.table['description']:
            available_sectors.append(int(re.search(r'\d+', sector).group()))

        return available_sectors
    
    def from_lightkurve(self, sectors=None, flatten=True, n_pca=5, aperture_mask=None, **kwargs):
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

        self.available_sectors = self.check_available_sectors()

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
            search_result_mask = np.ones(len(self.available_sectors), dtype=bool)
            masked_search_result = self.search_result
        
        # HACK for PHT
        # search_result_mask = np.ones(len(self.available_sectors), dtype=bool)
        # for sector in self.available_sectors:
        #     if sector >= 27 and sector < 56:
        #         search_result_mask.append(True)
        #     else:
        #         search_result_mask.append(False)

        masked_search_result = self.search_result[search_result_mask]

        # download data
        tpfc = lk.TargetPixelFileCollection([])
        for search_row in masked_search_result:
            try:
                tpfc.append(search_row.download(cutout_size=11))
            except:
                continue

        self.tpfc = tpfc
        self.model_lcc = lk.LightCurveCollection([])

        self.link_mask = []
        self.lcc = lk.LightCurveCollection([])
        self.breakpoints = []
        self.used_sectors = []

        lc = None
        
        # apply the pca background correction
        for tpf in tpfc:
            if aperture_mask is None:
                # define threshold aperture mask
                aperture_mask = tpf._parse_aperture_mask('threshold')
                if np.sum(aperture_mask) == 0:
                    aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
                    aperture_mask[round(tpf.shape[1]/2)-2:round(tpf.shape[1]/2)+1, \
                                  round(tpf.shape[2]/2)-2:round(tpf.shape[2]/2)+1] = True
            elif aperture_mask == 'center':
                aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
                aperture_mask[round(tpf.shape[1]/2)-2:round(tpf.shape[1]/2)+1, \
                              round(tpf.shape[2]/2)-2:round(tpf.shape[2]/2)+1] = True
            try:
                new_lc = self.apply_pca_corrector(tpf, flatten=flatten, zero_point_background=True, 
                                                  aperture_mask=aperture_mask, n_pca=n_pca, pipeline_call=True)
                new_raw_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)

                # stitch together
                if lc is None:
                    self.tpf = tpf
                    self.aperture_mask = aperture_mask
                    lc = new_lc
                    raw_lc = new_raw_lc
                else:
                    lc = lc.append(new_lc)
                    raw_lc = raw_lc.append(new_raw_lc)

                self.breakpoints.append(new_lc.time[-1])
                self.used_sectors.append(tpf.sector)
                self.lcc.append(new_lc)
            except:
                continue

        self.lc = lc
        self.raw_lc = raw_lc
        self.link_mask = np.concatenate(self.link_mask)

        return lc
    
    def from_local_data_mendel(self, sectors=None, aperture_mask='center', flatten=False, n_pca=5, **kwargs):
        """
        Retrieve data from local FFI data on Mendel.

        Parameters
        ----------
        sectors : int, list of ints
            desired sector number or list of sector numbers
        flatten : bool
            optionally flatten the light curve

        Returns
        -------
        lc : `lightkurve.LightCurve` object
            background-corrected flux time series
        """
        import astrocut

        if sectors is None:
            sectors = self.available_sectors

        tpfs = []

        ffi_path = '/shared_data/osn/astro-tessdata/ffi/'

        for i, sector in enumerate(sectors):
            cam = self.cameras[i]
            ccd = self.ccds[i]

            fits_image_paths = glob.glob(os.path.join(ffi_path, f's{sector:04}/*/*/{cam}-{ccd}/*.fits'))
            fits_image_paths.sort()

            try:
                my_cutter = astrocut.CutoutFactory()

                out_path = f'/home/nsaunders/mendel-nas1/cutout_files/tic{self.ticid}'
                cutout_file = my_cutter.cube_cut(f's3://stpubdata/tess/public/mast/tess-s{sector:04}-{cam}-{ccd}-cube.fits', 
                                                 self.coords, 11, output_path=out_path)

                tpf = lk.read(cutout_file, targetid=f'TIC {self.ticid}')

                file_to_remove = pathlib.Path(cutout_file)
                file_to_remove.unlink()
                tpfs.append(tpf)

            except:
                continue

        tpfc = lk.TargetPixelFileCollection(tpfs)

        self.tpfc = tpfc
        self.tpf = tpfc[0]

        self.model_lcc = lk.LightCurveCollection([])

        self.link_mask = []
        self.lcc = lk.LightCurveCollection([])
        self.breakpoints = []
        self.used_sectors = []
    
        lc = None

        # apply the pca background correction
        for tpf in tpfc:
            if aperture_mask is None:
                # define threshold aperture mask
                aperture_mask = tpf._parse_aperture_mask('threshold')
                if np.sum(aperture_mask) == 0:
                    aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
                    aperture_mask[round(tpf.shape[1]/2)-2:round(tpf.shape[1]/2)+1, \
                                round(tpf.shape[2]/2)-2:round(tpf.shape[2]/2)+1] = True
            elif aperture_mask == 'center':
                aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
                aperture_mask[round(tpf.shape[1]/2)-2:round(tpf.shape[1]/2)+1, \
                              round(tpf.shape[2]/2)-2:round(tpf.shape[2]/2)+1] = True
            try:
                new_lc = self.apply_pca_corrector(tpf, flatten=flatten, zero_point_background=True, 
                                                    aperture_mask=aperture_mask, n_pca=n_pca, pipeline_call=True)
                new_raw_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)

                # stitch together
                if lc is None:
                    self.tpf = tpf
                    self.aperture_mask = aperture_mask
                    lc = new_lc
                    raw_lc = new_raw_lc
                else:
                    lc = lc.append(new_lc)
                    raw_lc = raw_lc.append(new_raw_lc)

                self.breakpoints.append(new_lc.time[-1])
                self.used_sectors.append(tpf.sector)
                self.lcc.append(new_lc)
            except:
                continue

        self.lc = lc
        self.raw_lc = raw_lc
        self.link_mask = np.concatenate(self.link_mask)

        return lc

    
    def from_local_data(self, local_data_path, sectors=None, flatten=False, zero_point_background=False, n_pca=5):
        """
        Retrieve data from local data cube on Vanir.
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
        from astrocut import CutoutFactory

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
        lc = self.apply_pca_corrector(self.tpf, flatten=flatten, zero_point_background=zero_point_background, 
                                      aperture_mask=None, n_pca=n_pca)


        # store as LCC for plotting later
        self.lcc = lk.LightCurveCollection([lc])
        self.breakpoints = [lc.time[-1]]
        for tpf in tpfc[1:]:
            new_lc = self.apply_pca_corrector(tpf, flatten=flatten, zero_point_background=zero_point_background, 
                                              aperture_mask=None, n_pca=n_pca)
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
    
    def apply_pca_corrector(self, tpf, flatten=True, zero_point_background=False, aperture_mask=None, n_pca=5, pipeline_call=False):
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

        if aperture_mask is None:
            # define threshold aperture mask
            aperture_mask = tpf._parse_aperture_mask('threshold')
            if np.sum(aperture_mask) == 0:
                aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
                aperture_mask[round(tpf.shape[1]/2)-2:round(tpf.shape[1]/2)+1, \
                              round(tpf.shape[2]/2)-2:round(tpf.shape[2]/2)+1] = True
        elif aperture_mask == 'center':
            aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
            aperture_mask[round(tpf.shape[1]/2)-2:round(tpf.shape[1]/2)+1, \
                          round(tpf.shape[2]/2)-2:round(tpf.shape[2]/2)+1] = True

        # create boolean mask for tpf
        link_mask = np.ones_like(tpf.time.value, dtype=bool)
        
        # add the first 24 and last 12 hours of data to mask
        try:
            link_mask[tpf.time.value < tpf.time.value[0] + 1.0] = False
            link_mask[tpf.time.value > tpf.time.value[-1] - 0.5] = False
        except:
            link_mask = link_mask

        # identify the largest gap in the data
        try:
            gap = np.argmax(np.diff(tpf.time.value))

            # mask 24 hours after and 12 hours before the largest gap
            link_mask[(tpf.time.value < tpf.time.value[gap] + 1.0) & (tpf.time.value > tpf.time.value[gap] - 0.5)] = False
        except:
            link_mask = link_mask

        # drop False indicies from tpf
        tpf = tpf[link_mask]

        # create raw light curve            
        raw_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)

        # remove NaNs and negative flux values
        mask = (raw_lc.flux_err > 0) | (~np.isnan(raw_lc.flux))
        raw_lc = raw_lc[mask]
        tpf = tpf[mask]

        # create design matrix from pixels outside of aperture
        regressors = tpf.flux[:, ~aperture_mask]
        dm = lk.DesignMatrix(regressors, name='regressors')

        # perform PCA on design matrix and append column of constants
        dm = dm.pca(n_pca)
        dm = dm.append_constant()

        # fit weights to design matrix and remove background noise model
        corrector = lk.RegressionCorrector(raw_lc.normalize())
        corrected_lc_unnormalized = corrector.correct(dm)
        model = corrector.model_lc

        if pipeline_call:
            self.model_lcc.append(model)

        # optionally normalize to the 5th percentile of model flux
        if zero_point_background:
            model -= np.percentile(model.flux, 5)

        corrected_lc = lk.LightCurve(time=model.time, flux=raw_lc.normalize().flux.value-model.flux.value, flux_err=raw_lc.flux_err.value)

        if flatten:
            if tpf.sector <= 27:
                flatten_window = 501
            elif tpf.sector <= 56:
                flatten_window = 1501
            else:
                flatten_window = 4501
            corrected_lc = corrected_lc.flatten(flatten_window)

        if pipeline_call:
            self.link_mask.append(link_mask)

        corrected_lc.flux = corrected_lc.flux - 1.

        return corrected_lc
    
    def fetch_and_clean_data(self, lc_source='lightkurve', sectors=None, aperture_mask=None, flatten=True, zero_point_background=True, n_pca=5, **kwargs):
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
            lc = self.from_lightkurve(sectors=sectors, flatten=flatten, aperture_mask=aperture_mask, zero_point_background=zero_point_background, n_pca=n_pca)

        elif lc_source == 'local':
            lc = self.from_local_data('/data/users/nsaunders/cubes')

        elif lc_source == 'mendel':
            lc = self.from_local_data_mendel(sectors=sectors, aperture_mask=aperture_mask, flatten=flatten, n_pca=5)

        lc = self._clean_data(lc)

        return self
    
    def _clean_data(self, lc):
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
            self.fetch_and_clean_data(lc_source='lightkurve', sectors=s)

            fname_corr = f'{self.ticid}_s{s:02d}_corr.fits'
            fname_raw = f'{self.ticid}_s{s:02d}_raw.fits'

            path_corr = os.path.join(outdir, fname_corr)
            path_raw = os.path.join(outdir, fname_raw)

            self.lc.flux += 1.

            self.lc.to_fits(path=path_corr, overwrite=True)
            self.raw_lc.to_fits(path=path_raw, overwrite=True)

    def create_summary_plot(self, **kwargs):
        plot_summary(self, **kwargs)