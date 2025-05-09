import giants
import time
target = giants.Target(ticid=176956893)
print(f'Fetching and cleaning data...')
start = time.time()
target.fetch_and_clean_data(sectors=[1,2])
print(f'Fetch and clean complete. Total time: {time.time() - start}s')
print(f'Creating summary plot...')
start = time.time()
target.create_summary_plot()
print(f'Plot creation coe. Total time: {time.time() - start}s')