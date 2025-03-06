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
   ror: 00hx57361
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

When a planet orbiting a star outside of our solar system (known as an exoplanet) passes in front of its host star along our line of sight, it occults a fraction of the stellar light. This phenomenon is known as a transit. The *Transiting Exoplanet Survey Satellite* (TESS) is a NASA mission currently performing a nearly all-sky survey to search for transiting exoplanets [@ricker2018], which has already resulted in over 7,000 planet candidates and 600 confirmed planets [@nexsci]. TESS observes the majority of stars in large 24x90 degree rectangular regions of the sky known as Full-Frame Images (FFIs). These FFIs were observed at a 30-minute cadence for the TESS prime mission, 10-minute cadence for the first extended mission, and 200-second cadence in the second extended mission. We present the `giants` pipeline, a Python package to analyze TESS FFI observations to search for planets transiting evolved (subgiant and red giant) stars.

# Statement of Need

As a host star evolves into a subgiant and then a red giant, it becomes cooler, larger, and more luminous. This stellar evolution has two primary effects on a planet's transit---the transit becomes shallower and lower signal to noise due to the increased brightness of the host star it is transiting, and the duration of the transit increases as the stellar radius grows larger. Additionally, evolved stars exhibit photometric variability due to surface granulation which can obscure a transit signal [@cite]. 

Multiple large-scale transit-search efforts exist to identify new transiting planets in the TESS FFI observations, most prominently the MIT Quick-Look Pipeline (QLP; [@qlp]) and NASA's Science Processing Operations Center (SPOC, [@spoc]). These search efforts are intended to apply broadly to all transiting planet systems, ... 

# Data Access

The `giants` pipeline uses TESS FFI data stored on the Mikulski Archive for Space Telescopes (MAST; [@mast]). There are two primary methods for accessing these data: using the `TESSCut` implementation in the `lightkurve` Python package, or accessing the data directly on the MAST-hosted cloud storage.

## `TESSCut`


## Cloud Data

# Noise Removal

We utilize the `RegressionCorrector` module of the Lightkurve 

# Transiting Planet Search

# Acknowledgements

# References