# -*- coding: utf-8 -*-
"""
Multiscale (coarse-to-fine) driver for near-field ptychography.

Why
---
Projection algorithms (DM/HIO/ER) behave as *smoothers*: they reduce high
spatial-frequency error quickly and low-frequency error very slowly, because the
per-iteration convergence factor of a mode scales like 1 - c*(dx/wavelength)^2 --
near 1 for long-wavelength (low-frequency) modes. This is exactly why binning the
data speeds up low-frequency recovery in near-field ptychography: on a coarser
grid a fixed *physical* low frequency occupies a larger fraction of the Nyquist
band, so the smoother attacks it ~bin^2 faster. (There is also a secondary SNR
gain: binning photon-averages the gently-varying low-frequency signal.)

Plain binning throws away resolution. This driver instead runs a coarse-to-fine
cascade: solve on strongly-binned data (low frequencies converge in tens of
iterations), upsample the object and probe, and continue on the next finer grid,
which now only has to fix the high frequencies -- its fast direction. You keep
full final resolution *and* fast low-frequency convergence.

Usage
-----
    from ptypy.custom.nf_multiscale import run_multiscale
    # p is a fully-configured Ptycho parameter tree with one engine
    P = run_multiscale(p, bins=(4, 2, 1), iters=(40, 30, 60))

The same engine parameters (e.g. DMNearfield with its preconditioner / LF
constraint / probe anchor) are used at every level.

authors: J. C. da Silva
"""
import copy
import numpy as np
from scipy import ndimage as ndi

import ptypy
from ptypy.utils.verbose import log, headerline


def _resize_complex(arr, target_hw, order=1):
    """
    Resize a complex array's last two axes to `target_hw`, preserving the
    (centered) physical field of view. Real and imaginary parts are interpolated
    separately. Used to upsample object/probe between cascade levels.
    """
    sh = arr.shape
    src_hw = sh[-2:]
    if tuple(src_hw) == tuple(target_hw):
        return arr.copy()
    zoom = (target_hw[0] / src_hw[0], target_hw[1] / src_hw[1])
    full_zoom = (1.0,) * (arr.ndim - 2) + zoom
    re = ndi.zoom(arr.real, full_zoom, order=order)
    im = ndi.zoom(arr.imag, full_zoom, order=order)
    out = (re + 1j * im).astype(arr.dtype)
    # ndi.zoom can be off by a pixel; center-crop/pad to the exact target.
    return _center_fit(out, target_hw)


def _center_fit(arr, target_hw):
    """Center-crop or zero-pad the last two axes of `arr` to `target_hw`."""
    out = arr
    for ax, tgt in zip((-2, -1), target_hw):
        cur = out.shape[ax]
        if cur == tgt:
            continue
        if cur > tgt:  # crop
            start = (cur - tgt) // 2
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(start, start + tgt)
            out = out[tuple(sl)]
        else:          # pad
            before = (tgt - cur) // 2
            after = tgt - cur - before
            pad = [(0, 0)] * out.ndim
            pad[ax] = (before, after)
            out = np.pad(out, pad, mode='edge')
    return out


def _extract(P):
    """Snapshot object and probe storage arrays from a finished Ptycho."""
    obj = {name: s.data.copy() for name, s in P.obj.storages.items()}
    pr = {name: s.data.copy() for name, s in P.probe.storages.items()}
    return {'obj': obj, 'probe': pr}


def _inject(P, prev):
    """
    Seed the level's object and probe storages with the upsampled result of the
    previous (coarser) level. Storages are matched by name; modes are matched up
    to the smaller mode count.
    """
    for name, s in P.obj.storages.items():
        src = prev['obj'].get(name)
        if src is None:
            continue
        s.data[:] = _resize_complex(src, s.data.shape[-2:])
    for name, s in P.probe.storages.items():
        src = prev['probe'].get(name)
        if src is None:
            continue
        nm = min(src.shape[0], s.data.shape[0])
        s.data[:nm] = _resize_complex(src[:nm], s.data.shape[-2:])


def run_multiscale(p, bins=(4, 2, 1), iters=None, engine_label=None,
                   interp_order=1):
    """
    Run a coarse-to-fine near-field reconstruction.

    Parameters
    ----------
    p : ptypy.utils.Param
        A fully-configured Ptycho parameter tree (scans + one or more engines).
    bins : sequence of int
        Detector rebin factors, coarse to fine. Must divide the detector frame
        size; the last is usually 1 (full resolution).
    iters : sequence of int, optional
        Number of iterations per level. Defaults to scaling the engine's
        configured numiter as bin^2 (coarse levels are cheap and converge the
        low frequencies fast). Length must match `bins` if given.
    engine_label : str, optional
        Which engine's numiter to override per level. Defaults to the first.
    interp_order : int
        Spline order for the inter-level object/probe upsampling (1 = linear).

    Returns
    -------
    Ptycho
        The final (full-resolution) Ptycho instance, run and finalized.
    """
    if engine_label is None:
        engine_label = sorted(p.engines.keys())[0]
    base_iters = p.engines[engine_label].numiter
    if iters is None:
        iters = [int(base_iters * (b ** 2)) for b in bins]
    assert len(iters) == len(bins), "iters must match bins in length"

    prev = None
    P = None
    for level, (b, nit) in enumerate(zip(bins, iters)):
        log(3, '\n' + headerline(
            'Multiscale level %d/%d: rebin=%d, iters=%d'
            % (level + 1, len(bins), b, nit), 'c'))

        pl = copy.deepcopy(p)
        for scan in pl.scans.values():
            scan.data.rebin = b
        pl.engines[engine_label].numiter = nit

        # Initialise containers without running the engines, so we can seed
        # the object/probe from the previous (coarser) level first.
        P = ptypy.core.Ptycho(pl, level=4)
        if prev is not None:
            _inject(P, prev)
            log(3, 'Multiscale: seeded object/probe from previous level.')
        P.run()
        P.finalize()

        prev = _extract(P)

    return P
