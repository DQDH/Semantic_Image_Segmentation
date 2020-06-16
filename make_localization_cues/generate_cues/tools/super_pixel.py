import numpy as np
from scipy.ndimage import zoom
from skimage.segmentation import slic
def sup(init_cue,orig_img):
    img = zoom(orig_img, (41.0 / orig_img.shape[0], 41.0 / orig_img.shape[1], 1), order=1)
    segments = slic(img, n_segments=300, compactness=10)
    segments_id_all = np.unique(segments)
    for segments_id in segments_id_all:
        segments_region_id = np.where(segments == segments_id)
        segments_region_id_c = np.unique(init_cue[segments_region_id[0], segments_region_id[1]])
        if len(segments_region_id_c) == 2 and segments_region_id_c[1] == 22:
            init_cue[segments_region_id[0], segments_region_id[1]] = segments_region_id_c[0]
    return init_cue