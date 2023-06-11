import os
import time

dirpath = '/home/nsaunders/.lightkurve-cache/tesscut/'

while True:
    time.sleep(60)
    for f in os.listdir(dirpath):
        fn = os.path.join(dirpath, f)
        if os.stat(fn).st_mtime < time.time() - 60:
            try:
                os.remove(fn)
                print(f'Removed {f}')
            except:
                print(f'Could not remove {f}')