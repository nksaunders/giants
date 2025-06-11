from matplotlib.colors import same_color
import giants
import time
import numpy as np

import matplotlib.pyplot as plt

tics = [256722647, 176956893, 365102760, 348835438, 230001847, 219854519, 394918211, 204650483, 365102760, 290131778, 
        441462736, 257527578, 44792534,	348835438, 70524163, 29857954, 229510866, 200723869, 219854185, 258920431]

for tic in tics:
    target = giants.Target(ticid=tic)

    print(f'Fetching and cleaning data...')
    start = time.time()
    sectors = target.available_sectors
    print(sectors)
    target.fetch_and_clean_data(sectors=sectors)
    print(f'Fetch and clean omplete. Total time: {time.time() - start}s')
    print(f'Creating summary plot...')
    start = time.time()
    target.create_summary_plot()

    #plt.plot(target.lc.time.value, target.lc.flux_err.value)
    #plt.show()
    print(f'Plot creation coe. Total time: {time.time() - start}s')