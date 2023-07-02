import pandas as pd
import numpy as np
import os

def create_batch_file(inlist, outdir, batchfile_path, local=False):

    # read in the list of targets
    f = pd.read_csv(inlist, delimiter=',')
    targets = np.array(f['tic'])
    if len(targets) == 0:
        targets = np.array(f['TIC'])

    # create the batch file
    for tic in targets:
        if local:
            command = f'python run_giants.py {tic} {outdir} --local \n'
        else:
            command = f'python run_giants.py {tic} {outdir} \n'
        with open(f'{batchfile_path}.tot', 'a+') as file:
            file.write(command)

    # create corresponding directories
    os.mkdir(f'{outdir}/timeseries')
    os.mkdir(f'{outdir}/fft')
    os.mkdir(f'{outdir}/plots')

    # create the transit stats file
    with open(os.path.join(outdir, "transit_stats.txt"), "a+") as file:
        file.write("ticid depth depth_snr period t0 dur scaled_residuals\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a batch file to run giants on a list of targets.')
    parser.add_argument('inlist', type=str, help='Path to a list of TICIDs to run giants on.')
    parser.add_argument('outdir', type=str, help='Path to the output directory.')
    parser.add_argument('batchfile_name', type=str, help='Name of the batch file to be created.')
    parser.add_argument('--local', action='store_true', help='Flag to indicate whether to use local data.')
    args = parser.parse_args()

    create_batch_file(args.inlist, args.outdir, args.batchfile_name, args.local)