from matplotlib.colors import same_color
import giants
import time
import numpy as np

import matplotlib.pyplot as plt


target = giants.Target(ticid=256722647)

print(f'Fetching and cleaning data...')
start = time.time()
sectors = target.available_sectors
print(sectors)
# sectors = [52,57,58,77,78]: Correct period, incorrect radius
# adding 17,18,25: Incorrect period (harmonic of correct)
#sectors=[25,52,57,58,78,84,85,86]
target.fetch_and_clean_data(sectors=sectors)
print(f'Fetch and clean omplete. Total time: {time.time() - start}s')
print(f'Creating summary plot...')
start = time.time()
target.create_summary_plot()

#plt.plot(target.lc.time.value, target.lc.flux_err.value)
#plt.show()
print(f'Plot creation coe. Total time: {time.time() - start}s')