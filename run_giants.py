import os
import sys
import argparse

from giants.plotting import plot_summary
from giants.target import Target


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run giants on a single target.')
    parser.add_argument('ticid', type=str, help='TICID of the target to run giants on.')
    parser.add_argument('outdir', type=str, help='Path to the output directory.')
    parser.add_argument('--cloud', action='store_true', help='Flag to indicate whether to use cloud data.')
    parser.add_argument('--save_data', action='store_true', help='Flag to indicate whether to save the data.')
    args = parser.parse_args()

    try:
        if os.path.isfile(os.path.join(args.outdir, '/timeseries/{args.ticid}.dat.ts')): 
            print(f'TIC {args.ticid} already exists.')

        else:
            target = Target(ticid=args.ticid)

            if args.cloud:
                target.fetch_and_clean_data(lc_source='cloud')
            else:
                target.fetch_and_clean_data(lc_source='lightkurve')
            plot_summary(target, outdir=args.outdir, save_data=args.save_dataa)

    except:
        print(f'Target {args.ticid} failed.')
