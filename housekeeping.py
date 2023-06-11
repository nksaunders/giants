import os
import time

while True:
    time.sleep(60)
    for f in os.listdir('/home/nsaunders/.lightkurve-cache/tesscut/'):
        if os.stat(f).st_mtime < time.time() - 60:
            os.remove(f)