# Third Feature - depth through correspoinding points of shadows and objects intersecting at a single point
# Checks if shadows are consistent with their matching objects and created by the same light source to verify authenticity

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import io, contextlib
from typing import Tuple, List, Optional

# use same shadow mask as texture.py
from texture import make_shadow_mask as texture_make_shadow_mask

def mask_from_texture(img_bgr, suppress_prints=True, **kwargs): 
    # call texture.py's make_shadow_mask function to get the shadow mask
    if suppress_prints:
        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink):
            out = texture_make_shadow_mask(img_bgr, **kwargs)
    else:
        out = texture_make_shadow_mask(img_bgr, **kwargs)

    if isinstance(out, tuple):
        mask = out[0]
    else:
        mask = out
    return mask