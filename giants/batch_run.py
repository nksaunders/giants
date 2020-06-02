"""
Script to run a batch of TIC IDs.
"""

import sys
try:
    from .giants import Giant
except:
    from giants import Giant


def save_fits_file(ticid):
    try:
        target = Giant(ticid=ticid, csv_path='data/ticgiants_allsky_halo.csv',
                         cache_path='/data/sarek1/nksaun/lightkurve_cache')

        target.save_to_fits(outdir='/data/sarek1/nksaun/tess_giants/tayar_lcs')

    except:
        try:
            target = Giant(ticid=ticid, csv_path='data/ticgiants_allsky_halo.csv',
                             cache_path='/data/sarek1/nksaun/lightkurve_cache')

            target.save_to_fits(outdir='/data/sarek1/nksaun/tayar_giants/zinn_lcs')

        except:
            try:
                target = Giant(ticid=ticid, csv_path='data/ticgiants_allsky_halo.csv',
                                 cache_path='/data/sarek1/nksaun/lightkurve_cache')

                target.save_to_fits(outdir='/data/sarek1/nksaun/tayar_giants/zinn_lcs')
            except:
                pass


if __name__ == '__main__':
    try:
        ticid = sys.argv[1]
        print(f'Generating fits files for target {ticid}...')

        save_fits_file(ticid)
    except:
        print(f'Failed to save fits files for target {ticid}.')
