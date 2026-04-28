import sys
import re
from io import BytesIO
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import mmcv
import numpy as np
from numpy import ndarray


def visualize_disp(disp: np.ndarray, save_file: str = None,
                   model_name: str = None, runtime: str = None, mask: np.ndarray=None) -> np.ndarray:
    """Disparity visualization function.

    Args:
        disp (ndarray): The disparity will be render
        save_dir ([type], optional): save dir. Defaults to None.
    Returns:
        ndarray: disparity map image with RGB order.
    """
    # return value from mmcv.flow2rgb is [0, 1.] with type np.float32
    disp = disp.squeeze()
    max_disp = max(disp.max(), 5)
    flo = disp/max_disp
    flo[flo > 1.0] = 1.0
    flo[flo < 0.01] = 0.01
    flo = 1.0 - flo
    B = flo * 255
    R = 255 - B
    G = 255 - np.fabs(flo - 0.5) * 2 * 255
    merged = cv2.merge([B, G, R])
    if mask is not None:
        merged[mask] = 0
    disp_map = merged.astype(np.uint8)

    if model_name:
        cv2.putText(disp_map, model_name, (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    if runtime:
        cv2.putText(disp_map, runtime, (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    if save_file:
        # plt.imsave(save_file, disp_map)
        cv2.imwrite(save_file, disp_map)
    return disp_map

def visualize_depth_contour(depth: np.ndarray, max_depth: float, save_file: str = None,
                                  model_name: str = None, runtime: str = None, mask: np.ndarray = None,
                                  contour_interval: float = 5.0) -> np.ndarray:
    """Enhanced depth visualization with colorbar and contour lines.

    Args:
        depth (ndarray): The depth data to visualize
        max_depth (float): Maximum depth value for normalization
        save_file (str, optional): save file path. Defaults to None.
        model_name (str, optional): model name to display. Defaults to None.
        runtime (str, optional): runtime info (e.g. cam_params) to display. Defaults to None.
        mask (ndarray, optional): mask to apply. Defaults to None.
        contour_interval (float): Interval for contour lines in meters. Defaults to 5.0.
    Returns:
        ndarray: depth map image with RGB order.
    """
    depth = depth.squeeze()
    
    # Get original image dimensions for consistent output size
    height, width = depth.shape[:2]
    dpi = 200.0
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Create subplot with no margins
    ax = fig.add_subplot(111)
    ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height] in figure coordinates
    
    # Create depth visualization
    depth_display = depth.copy()
    depth_display[depth > max_depth] = max_depth

    # Invert to match cv2.COLORMAP_JET
    depth_norm = depth_display / max_depth
    depth_norm[depth_norm > 1] = 1
    depth_norm[depth_norm < 0.01] = 0.01
    depth_norm = 1 - depth_norm 
    depth_display = depth_norm * max_depth

    # Use jet colormap for depth visualization
    norm = Normalize(vmin=0, vmax=max_depth)
    cmap = matplotlib.colormaps.get_cmap('jet')
    
    # Display depth image
    im = ax.imshow(depth_display, cmap=cmap, norm=norm)
    if mask is not None:
        # Create a masked array to hide masked regions
        masked_depth = np.ma.masked_array(depth_display, mask=mask)
        im = ax.imshow(masked_depth, cmap=cmap, norm=norm)
    ax.axis('off')  # Remove all axes decorations
    
    # Add contour lines at specified intervals
    if contour_interval > 0:
        # Use original depth values for contour levels and labels
        original_depth = depth.copy()
        original_depth[depth > max_depth + 0.01] = max_depth + 0.01

        # Generate contour levels, ensure max_depth level is included
        contour_levels = np.arange(0, max_depth + contour_interval, contour_interval)

        conCmap = matplotlib.colormaps.get_cmap('jet')
        contour_colors = [conCmap(min(level / max_depth, 0.99)) for level in contour_levels]

        contours = ax.contour(original_depth, levels=contour_levels,
                              colors=contour_colors, linewidths=1.2, alpha=0.9)
        ax.clabel(contours, inline=True, fontsize=9, fmt='%dm', colors=contour_colors)
    
    # Add text overlays
    text_y = 0.95
    if model_name:
        ax.text(0.02, text_y, model_name, transform=ax.transAxes, 
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        text_y -= 0.08
    
    if runtime:
        ax.text(0.02, text_y, runtime, transform=ax.transAxes, 
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        text_y -= 0.08
    
    # Add depth threshold info
    ax.text(0.02, text_y, f'Max Depth: {max_depth}m', transform=ax.transAxes, 
            fontsize=8, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # Save the figure if requested
    if save_file:
        plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                   facecolor='white', edgecolor='none')
        plt.close()
        img = cv2.imread(save_file)
        return img
    else:
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
def visualize_depth(depth: np.ndarray, max_depth: float,save_file: str = None,
                    model_name: str = None, runtime: str = None, mask: np.ndarray=None) -> np.ndarray:
    """Disparity visualization function.

    Args:
        depth (ndarray): The disparity will be render
        save_dir ([type], optional): save dir. Defaults to None.
    Returns:
        ndarray: disparity map image with RGB order.
    """
    # return value from mmcv.flow2rgb is [0, 1.] with type np.float32
    depth = depth.squeeze()
    depth_norm = depth / max_depth
    depth_norm[depth_norm > 1] = 1
    depth_norm[depth_norm < 0.01] = 0.01
    depth_norm = 1 - depth_norm
    color_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_norm, alpha=270), cv2.COLORMAP_JET)
    if mask is not None:
        color_depth[mask] = 0
    depth_map = color_depth.astype(np.uint8)

    if model_name:
        cv2.putText(depth_map, model_name, (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    if runtime:
        cv2.putText(depth_map, runtime, (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    if save_file:
        cv2.imwrite(save_file, depth_map)
    return depth_map

# problematic function, actually for flow instead of disparity
def read_disp_kitti(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read sparse disp file from KITTI dataset.

    This function is modified from
    https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py.
    Copyright (c) 2020, princeton-vl
    Licensed under the BSD 3-Clause License

    Args:
        name (str): The disp file

    Returns:
        Tuple[ndarray, ndarray]: disp and valid map
    """
    # to specify not to change the image depth (16bit)
    disp = cv2.imread(name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    disp = disp[:, :, ::-1].astype(np.float32)
    # disp shape (H, W, 2) valid shape (H, W)
    disp, valid = disp[:, :, :2], disp[:, :, 2]
    disp = (disp - 2**15) / 64.0
    return disp, valid

# problematic function, actually for flow instead of disparity
def write_disp_kitti(uv: np.ndarray, filename: str):
    """Write the disp in disk.

    This function is modified from
    https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py.
    Copyright (c) 2020, princeton-vl
    Licensed under the BSD 3-Clause License

    Args:
        uv (ndarray): The disparity that will be saved.
        filename ([type]): The file for saving disparity.
    """
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def disp_from_bytes(content: bytes, suffix: str = 'flo') -> ndarray:
    """Read dense disparity from bytes.

    .. note::
        This load disparity function works for FlyingChairs, FlyingThings3D,
        Sintel, FlyingChairsOcc datasets, but cannot load the data from
        ChairsSDHom.

    Args:
        content (bytes): Disparity bytes got from files or other streams.

    Returns:
        ndarray: Loaded disparity with the shape (H, W, 2).
    """

    assert suffix in ('flo', 'pfm'), 'suffix of disp file must be `flo` '\
        f'or `pfm`, but got {suffix}'

    if suffix == 'flo':
        return flo_from_bytes(content)
    else:
        return pfm_from_bytes(content)


def flo_from_bytes(content: bytes):
    """Decode bytes based on flo file.

    Args:
        content (bytes): Disparity bytes got from files or other streams.

    Returns:
        ndarray: Loaded disparity with the shape (H, W, 2).
    """

    # header in first 4 bytes
    header = content[:4]
    if header != b'PIEH':
        raise Exception('Disp file header does not contain PIEH')
    # width in second 4 bytes
    width = np.frombuffer(content[4:], np.int32, 1).squeeze()
    # height in third 4 bytes
    height = np.frombuffer(content[8:], np.int32, 1).squeeze()
    # after first 12 bytes, all bytes are disp
    disp = np.frombuffer(content[12:], np.float32, width * height * 2).reshape(
        (height, width, 2))

    return disp


def pfm_from_bytes(content: bytes) -> np.ndarray:
    """Load the file with the suffix '.pfm'.

    Args:
        content (bytes): disparity bytes got from files or other streams.

    Returns:
        ndarray: The loaded data
    """

    file = BytesIO(content)

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.frombuffer(file.read(), endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    # return data[:, :, :-1]
    return data


def read_pfm(file: str) -> np.ndarray:
    """Load the file with the suffix '.pfm'.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        file (str): The file name will be loaded

    Returns:
        ndarray: The loaded data
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('ascii'))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('ascii').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    # return data[:, :, :-1]
    return data

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(str.encode('PF\n') if color else str.encode('Pf\n'))
    file.write(str.encode('%d %d\n' % (image.shape[1], image.shape[0])))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(str.encode('%f\n' % scale))

    image.tofile(file) 
