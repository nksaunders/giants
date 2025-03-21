import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
CACHEDIR = None # set this to a directory to cache data, defaults to ~/.giants_cache
if CACHEDIR is None:
    CACHEDIR = os.path.join(os.path.expanduser('~'), '.giants_cache')
os.makedirs(CACHEDIR, exist_ok=True)
CUTOUTDIR = os.path.join(CACHEDIR, 'cutouts')
os.makedirs(CUTOUTDIR, exist_ok=True)

from .target import *
from .plotting import *
from .utils import *