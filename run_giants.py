import sys
import argparse
import pandas as pd
import numpy as np

from giants.plotting import plot_summary
from giants.target import Target


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run giants on a single target.')
    parser.add_argument('ticid', type=str, help='TICID of the target to run giants on.')
    parser.add_argument('outdir', type=str, help='Path to the output directory.')
    parser.add_argument('--local', action='store_true', help='Flag to indicate whether to use local data.')
    args = parser.parse_args()

    try:
    
        if args.local:
            lookup_table = pd.read_csv('/data/users/sgrunblatt/TESS_targetlists/TIC_lookup.csv')

            N = np.nan
            for i in range(len(lookup_table)):
                if (int(args.ticid) > lookup_table['TIC_start'].iloc[i]) and (int(args.ticid) > lookup_table['TIC_start'].iloc[i]):
                    N = i
            if np.isnan(N):
                print(f'{args.ticid} not found in target lists!')
                csv_path = None
            else:
                csv_path = f'/data/users/sgrunblatt/TESS_targetlists/ticgiants_sublists/sublist{N}.csv'
        else:
            csv_path = None

        target = Target(ticid=args.ticid, csv_path=csv_path)

        target.fetch_and_clean_data(lc_source='local')
        plot_summary(target, outdir=args.outdir, save_data=True)
        target.save_to_fits(outdir=args.outdir)
    except:
        print(f'Target {sys.argv[1]} failed.')
