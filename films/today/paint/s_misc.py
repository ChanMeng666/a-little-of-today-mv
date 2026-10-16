"""Things that belong to no one set: the sheet of paper the whole film is printed on, (the tape under the lyric is in s_props.py)."""
import math

import numpy as np

from kit import *


def paper():
    paper_tile(512, 3)


ALL = {"paper": paper}
