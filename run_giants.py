import sys
import os
from giants.plotting import plot_summary
from giants.target import Target


if __name__ == '__main__':
    try:
        ticid = sys.argv[1]
        outdir = sys.argv[2]
        output = "plot"
        try:
            output = sys.argv[3]
        except:
            pass

        N = np.nan
        for i in range(len(lookup_table)):
            if (ticid > lookup_table['TIC_start'].iloc[i]) and (ticid > lookup_table['TIC_start'].iloc[i]):
                N = i
        if np.isnan(N):
            print(f'{ticid} not found in target lists!')
        else:
            csv_path = f'/data/users/sgrunblatt/TESS_targetlists/ticgiants_sublists/sublist{N}.csv'

        target = Target(ticid=ticid, csv_path=csv_path)

        if output=="plot":
            target.fetch_and_clean_data(lc_source='local')
            plot_summary(target, outdir=outdir, save_data=True)
        else:
            target.save_to_fits(outdir=outdir)
    except:
        print(f'Target {sys.argv[1]} failed.')
