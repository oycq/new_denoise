import glob
import os.path as osp
import warnings
import matplotlib
import numpy as np

def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    It will be used when automatically resume, modified from
    https://github.com/open-mmlab/mmdetection/blob/dev-v2.20.0/mmdet/utils/misc.py.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.

    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """ # noqa
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        # `count` is iteration number, as checkpoints are saved as
        # 'iter_xx.pth' or 'epoch_xx.pth' and xx is iteration number.
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

def colorMap(colormap_name: str, arr: np.ndarray,
             min_v=None, max_v=None, alpha=None) -> np.ndarray:
    arr = arr.cpu().numpy().astype(np.float64).squeeze()
    if colormap_name == 'oliver': return colorMapOliver(arr, min_v, max_v)
    cmap = matplotlib.cm.get_cmap(colormap_name)
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    arr[arr > max_v] = max_v
    arr[arr < min_v] = min_v
    arr = (arr - min_v) / (max_v - min_v + 1e-8) # avoid divided by 0
    if alpha is None:
        out = cmap(arr)
        out = out[:, :, 0:3]
    else:
        out = cmap(arr, alpha=alpha)
    return np.round(255 * out).astype(np.uint8)

#
# code adapted from Oliver Woodford's sc.m
_CMAP_OLIVER = np.array(
    [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174], [0,1,0,114],
     [0,1,1,185], [1,1,0,114], [1,1,1,0]]).astype(np.float64)
#
def colorMapOliver(arr: np.ndarray, min_v=None, max_v=None) -> np.ndarray:
    arr = arr.cpu().numpy().astype(np.float64).squeeze()
    height, width = arr.shape
    arr = arr.reshape([1, -1])
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    arr[arr < min_v] = min_v
    arr[arr > max_v] = max_v
    arr = (arr - min_v) / (max_v - min_v)
    bins = _CMAP_OLIVER[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = np.sum(
        np.tile(arr, [6, 1]) > \
        np.tile(np.reshape(cbins,[-1, 1]), [1, arr.size]), axis=0)
    ind[ind > 6] = 6
    bins = 1 / bins
    cbins = np.array([0.0] + cbins.tolist())
    arr = (arr - cbins[ind]) * bins[ind]
    arr = _CMAP_OLIVER[ind, :3] * np.tile(np.reshape(1 - arr,[-1, 1]),[1,3]) + \
        _CMAP_OLIVER[ind+1, :3] * np.tile(np.reshape(arr,[-1, 1]),[1,3])
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    out = np.reshape(arr, [height, width, 3])
    out = np.round(255 * out).astype(np.uint8)
    return out
