"""
Script to run a batch of TIC IDs.
"""

import sys
try:
    from .giants import Giant
    from .plotting import plot_summary
except:
    from giants import Giant
    from giants.plotting import plot_summary


def save_fits_file(ticid):

    csv_path = '/data/users/sgrunblatt/TESS_targetlists/ticgiants_allsky_halo.csv'
    cache_path = '/data/users/nsaunders/cubes'
    outdir = '/data/users/nsaunders/outputs'

    try:
        target = Giant(ticid=ticid, csv_path=csv_path,
                         cache_path=cache_path)

        # target.save_to_fits(outdir=outdir)
        target.fetch_and_clean_data(lc_source='local')
        plot_summary(target, save_fig=True, save_data=True, outdir=outdir)

    except:
        pass
        # try:
        #     target = Giant(ticid=ticid, csv_path=csv_path,
        #                      cache_path=cache_path)
        #
        #     # target.save_to_fits(outdir=outdir)
        #     target.fetch_and_clean_data(lc_source='local')
        #     plot_summary(target, save_fig=True, save_data=True, outdir=outdir)
        #
        # except:
        #     try:
        #         target = Giant(ticid=ticid, csv_path=csv_path,
        #                          cache_path=cache_path)
        #
        #         # target.save_to_fits(outdir=outdir)
        #         target.fetch_and_clean_data(lc_source='local')
        #         plot_summary(target, save_fig=True, save_data=True, outdir=outdir)
        #     except:
        #         pass


if __name__ == '__main__':
    try:
        ticid = sys.argv[1]
        print(f'Generating fits files for target {ticid}...')

        save_fits_file(ticid)
    except:
        print(f'Failed to save fits files for target {ticid}.')
