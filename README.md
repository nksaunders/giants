# the giants pipeline
---
an open-source Python package to search for exoplanets around evolved stars

**This package is currently under development. Please reach out if you experience any issues.**

## Read the papers here!
- [ADS Library](https://ui.adsabs.harvard.edu/public-libraries/AZJgdEbDRF66QORFZmL2Vw)

## To install the giants pipeline:
```python
!git clone https://github.com/nksaunders/giants.git
!cd giants; pip install .
```

To produce a de-trended light curve and a transit-search summary plot:

```python
import giants
target = giants.Target(ticid=176956893)
target.fetch_and_clean_data(sectors=[1,2])
target.create_summary_plot()
```

By default, this will query MAST and use all available sectors to produce a light curve (this can take a few minutes for some targets!). If you would like to specify which sectors to use, pass in a list of sectors numbers to the `fetch_and_clean_data` function, as shown above.

This will produce the following output:
![giants_demo](https://user-images.githubusercontent.com/17130840/166803594-5edc052f-663c-405d-b5d5-efbc86dbf06d.png)


You can also save the light curve for future use when producing the summary plot by passing in the `save_data=True` argument to `create_summary_plot`.
