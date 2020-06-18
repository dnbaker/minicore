import sys
import os

from selexp import *

def make_table(paths):
    assert all(map(os.path.isfile, paths))
