import numpy as np
from math import sqrt


def _split(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def _unsplit(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def split(img, chunks_num):
    chunk_dim = int(sqrt(chunks_num))
    min_dim = min(list(img.shape)[:-1])
    divisible_crop_dim = min_dim % chunk_dim
    cropped = img[:min_dim - divisible_crop_dim,:min_dim - divisible_crop_dim,:] # for classification it's better to loose some pixels, than add 0 that may be interpreted as dark water
    _, h, _ = cropped.shape
    new_dim = h // chunk_dim
    cropped_rearanged = np.moveaxis(cropped, -1, 0)
    chunks = []
    for channel in cropped_rearanged:
        chunks.append(_split(channel, new_dim, new_dim))
    stacked = np.stack(chunks, axis=1)
    stacked = np.moveaxis(stacked,1,-1)
    return stacked

def unsplit(split_imgs):
    chunks_num = split_imgs.shape[0]
    chunk_dim = int(sqrt(chunks_num))
    color_first = np.moveaxis(split_imgs,-1,0)
    chunks = []
    for channel in color_first:
        chunks.append(_unsplit(channel, channel.shape[1]*chunk_dim, channel.shape[2]*chunk_dim))
    stacked = np.stack(chunks, axis=0)
    stacked = np.moveaxis(stacked,0,-1)
    return stacked
