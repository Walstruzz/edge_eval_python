"""
Edge NMS
See https://github.com/pdollar/edges/blob/master/private/edgesNmsMex.cpp
"""

import numpy as np


def interp(image, h, w, x, y):
    x = min(max(0, x), w - 1.001)
    y = min(max(0, y), h - 1.001)
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    dx0, dy0 = x - x0, y - y0
    dx1, dy1 = 1 - dx0, 1 - dy0
    out = (image[y0, x0] * dx1 * dy1 +
           image[y0, x1] * dx0 * dy1 +
           image[y1, x0] * dx1 * dy0 +
           image[y1, x1] * dx0 * dy0)
    return out


def edge_nms(edge, ori, r, s, m):
    """
    CXX-like implementation
    """
    assert edge.ndim == 2 and ori.ndim == 2
    h, w = edge.shape
    out = np.empty((h, w), dtype=np.float32)
    for x in range(w):
        for y in range(h):
            e = out[y, x] = edge[y, x]
            if e == 0:
                continue
            e *= m
            cos_o = np.cos(ori[y, x])
            sin_o = np.sin(ori[y, x])
            for d in range(-r, r + 1):
                if d != 0:
                    e0 = interp(edge, h, w, x + d * cos_o, y + d * sin_o)
                    if e < e0:
                        out[y, x] = 0
                        break

    s = min(min(w // 2, s), h//2)
    for x in range(s):
        for y in range(h):
            out[y, x] *= 1.0 * x / s
            out[y, w - 1 - x] *= 1.0 * x / s
    for x in range(w):
        for y in range(s):
            out[y, x] *= 1.0 * y / s
            out[h - 1 - y, x] *= 1.0 * y / s
    return out


def fast_edge_nms(edge, ori, r, s, m):
    """
    Numpy Implementation, Faster!
    """
    assert edge.ndim == 2 and ori.ndim == 2
    h, w = edge.shape
    zero_mask = edge == 0
    out = edge.copy()
    cos_o, sin_o = np.cos(ori), np.sin(ori)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    e0 = np.zeros((h, w, r * 2), dtype=edge.dtype)
    cnt = 0
    for d in range(-r, r + 1):
        if d == 0:
            continue
        interp_x = xx + d * cos_o
        interp_y = yy + d * sin_o
        interp_x = np.clip(interp_x, 0, w - 1.001)
        interp_y = np.clip(interp_y, 0, h - 1.001)
        x0, y0 = interp_x.astype(int), interp_y.astype(int)
        x1, y1 = x0 + 1, y0 + 1
        dx0, dy0 = interp_x - x0, interp_y - y0
        dx1, dy1 = 1 - dx0, 1 - dy0
        e0[:, :, cnt] = (edge[y0, x0] * dx1 * dy1 + edge[y0, x1] * dx0 * dy1 +
                         edge[y1, x0] * dx1 * dy0 + edge[y1, x1] * dx0 * dy0)
        cnt += 1
    less_mask = (e0 < out[:, :, None] * m).min(axis=-1)
    mask = np.logical_or(zero_mask, less_mask)
    out = out * mask

    s = min(min(w // 2, s), h // 2)
    scale = 1.0 * np.arange(s) / s
    out[:, :s] *= scale
    out[:, -s:] *= scale[::-1]
    out[:s, :] *= scale[:, None]
    out[-s:, :] *= scale[::-1, None]
    return out







