---
title: 'The Giants Pipeline: A Python package to search for exoplanets orbiting evolved stars'
tags:
  - Python
  - astronomy
  - exoplanets
  - stellar evolution
authors:
  - name: Nicholas Saunders
    orcid: 0000-0003-2657-3889
    corresponding: true
    affiliation: "1, 4" 
  - name: Samuel K. Grunblatt
    orcid: 0000-0003-4976-9980
    affiliation: 2
  - name: Daniel Huber
    orcid: 0000-0001-8832-4488
    affiliation: 1
  - name: Emma Page
    orcid: 0000-0002-3221-3874
    afiliation: 3
affiliations:
 - name: Institute for Astronomy, University of Hawaiʻi at M\=anoa, 2680 Woodlawn Drive, Honolulu, HI 96822, USA
   index: 1
 - name: Department of Physics and Astronomy, The University of Alabama, 514 University Blvd., Tuscaloosa, AL 35487, USA
   index: 2
 - name: Department of Physics, Lehigh University, 16 Memorial Drive East, Bethlehem, PA 18015, USA
   index: 3
 - name: NSF Graduate Research Fellow
   index: 4
date: 05 February 2025
bibliography: paper.bib
---

# Summary

When a planet orbiting a star outside of our solar system (known as an exoplanet) passes in front of its host star along our line of sight, it occults a fraction of the stellar light. This phenomenon is known as a transit. The *Transiting Exoplanet Survey Satellite* (TESS) is a NASA mission currently performing a nearly all-sky survey to search for transiting exoplanets [@ricker2015], which has already resulted in over 7,000 planet candidates and 600 confirmed planets[^1]. TESS observes the majority of stars in large 24x90 degree rectangular regions of the sky known as Full-Frame Images (FFIs). These FFIs were observed at a 30-minute cadence for the TESS prime mission, 10-minute cadence for the first extended mission, and 200-second cadence in the second extended mission. We present the `giants` pipeline, a Python package to analyze TESS FFI observations to search for planets transiting evolved (subgiant and red giant) stars.

# Statement of Need

As a host star evolves into a subgiant and then a red giant, it becomes cooler, larger, and more luminous. This stellar evolution has two primary effects on a planet's transit---the transit becomes shallower and lower signal to noise due to the increased brightness of the host star it is transiting, and the duration of the transit increases as the stellar radius grows larger. Additionally, evolved stars exhibit photometric variability due to surface granulation which can obscure a transit signal [@cite]. 

Multiple large-scale transit-search efforts exist to identify new transiting planets in the TESS FFI observations, most prominently the MIT Quick-Look Pipeline (QLP; [@huang2020]) and NASA's Science Processing Operations Center (SPOC, [@jenkins2020]). These search efforts are intended to apply broadly to all transiting planet systems, ... 

# Data Access

The `giants` pipeline uses TESS FFI data stored on the Mikulski Archive for Space Telescopes (MAST) operated by the Space Telescope Science Institute. There are two primary methods for downloading these data: accessing the data directly on the MAST-hosted cloud storage, or using the `TESSCut` [@brasseur2019] implementation in the `lightkurve` Python package.

## Cloud Data

The MAST archive hosts image cubes containing TESS FFI observations on a publicly accessible cloud server using S3 Uniform Resource Identifers (URIs). The `giants` pipeline downloads pixel cutouts from these image cubes using the `from_cloud_data` method of the `giants.Target` object. This is the default and recommended method. 

## `TESSCut`

To download FFI observations with `TESSCut`, a user may call the `giants.Target` method `from_lightkurve`. This will use the `lightkurve.search_tesscut` function, which queries available observations and downloads TESS FFI pixel cutouts with the Python implementation of the `TESSCut` tool[^2]. 

# Noise Removal

We utilize the `RegressionCorrector` module of the Lightkurve to remove common sources of instrumental noise that inhibit transiting planet detection. The most significant contribution to the noise is produced by the light scattered onto the TESS detector by the Earth and Moon as TESS orbits, which manifests as highly periodic, high-amplitude spikes in the measured flux. For a small region of the detector (i.e. our 11x11 pixel cutouts), this signal is present in every pixel and broadly spatially uniform. 

To remove the scattered light background, we first define an aperture which captures the flux from the target star. By default, we use an aperture mask containing the central 3x3 pixels in the cutout; however, the aperture mask can also be set to `'threshold'` or `'pipeline'` following the functionality in `lightkurve`. We then create a design matrix of regressors, each containing the time-series flux of a single pixel outside of the target aperture, and perform principle component analysis (PCA) on the columns of the design matrix. We use PCA to reduce our design matrix to the five most significant signals shared among background pixels. The `RegressionCorrector` fits weights to the reduced design matrix to identify the contribution of the background regressors to the flux *within* the target aperture to construct a noise model, which is subtracted from the target flux, leaving the desired signal intact.

# Transiting Planet Search

We search for transiting planet signals using the `astropy` implementation of the `BoxLeastSquares` (BLS) algorithm. We first bin the data to 30-minute cadence for uniformity, and search over a grid of 5,000 periods evenly spaced between 2 and 30 days and 1,000 durations evenly spaced between 0.1 and 1.0 days. 

# Summary Plots

For each target, we produce a summary plot which contains information necessary to perform by-eye vetting. An example is shown in Figure \ref{fig:summary}. 

![Summary plot for TIC 139474683 produced by the `giants` pipeline.\label{fig:summary}](summary.png)

# Acknowledgements

The `giants` pipeline relies heavily on previous open source astronomical software packages, primarily `lightkurve` and `astropy`. We would like to thank the developers of these packages for providing high-quality and well-documented software tools to the astronomical community. N.S. acknowledges support by the National Science Foundation Graduate Research Fellowship Program under Grant Numbers 1842402 & 2236415 and NASA’S Interdisciplinary Consortia for Astrobiology Research (NNH19ZDA001N-ICAR) under award number 19-ICAR19 2-0041. 

# References

[^1]: Retrieved from the NASA Exoplanet Science Institute (NexScI; DOI:10.26133/NEA12) on 05 March 2025.

[^2]: https://mast.stsci.edu/tesscut/

[^3]: https://github.com/spacetelescope/astrocut