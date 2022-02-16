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

        target = Target(ticid=ticid)

        if output=="plot":
            target.fetch_and_clean_data(lc_source='local')
            plot_summary(target, outdir=outdir, save_data=True)
        else:
            target.save_to_fits(outdir=outdir)
    except:
        print(f'Target {sys.argv[1]} failed.')
