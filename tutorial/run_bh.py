#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver: bilinear-Hessian (BH-CG) reconstruction of near-field ptychography data
using ptypy.custom.bh_ptycho.BHPtycho.

Reads the Ecat near-field HDF5 (data, positions, geometry), builds a sensible
initial guess (Paganin phase for the object, back-propagated reference for the
probe), and runs BH-CG.

Backend:
  * CPU (NumPy)  -> set USE_GPU = False ; use BIN >= 2 so it runs in minutes.
  * GPU (CuPy)   -> set USE_GPU = True  ; BIN = 1 for full resolution.

Cluster note: develop on the laptop, then on the cluster just
    git pull && python run_bh.py
and paste the printed errors / saved PNGs back.
"""
import os
import sys
import time
import h5py
import numpy as np

# --- import the BH engine from the repo (works like ptycho_recons.py does) ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptypy.custom.bh_ptycho import BHPtycho

# ----------------------------- settings ----------------------------------- #
DATA = "Ecat_2nd_time_NFP_070nm_subtomo001_0000.h5"
# Defaults can be overridden from the shell with environment variables, so no
# file editing is needed on the cluster, e.g.:
#   BH_GPU=1 BH_BIN=1 BH_DELTA_BETA=500 BH_NITER=200 python run_bh.py
USE_GPU = bool(int(os.environ.get("BH_GPU", "0")))      # True on the cluster (CuPy)
BIN = int(os.environ.get("BH_BIN", "2"))                # 1 = full res (GPU), 4 = quick CPU test
NITER = int(os.environ.get("BH_NITER", "100"))
DELTA_BETA = float(os.environ.get("BH_DELTA_BETA", "300.0"))
# Paganin delta/beta for object init: MUST match your sample (soft/biological
# tissue at 33 keV ~ few hundred to ~1000; too small -> hollow object, as
# Nikitin's demo value 24 was). Use the value from your existing pipeline.
RHO = (1.0, 2.0)       # object/probe preconditioning scales (Appendix II)

if USE_GPU:
    import cupy as xp
    _dev = xp.cuda.Device()
    _free, _total = xp.cuda.runtime.memGetInfo()
    print("GPU backend: CuPy on device %d, free %.1f / %.1f GB"
          % (_dev.id, _free / 1e9, _total / 1e9), flush=True)
else:
    xp = np
    print("CPU backend: NumPy", flush=True)


def bin2(a, b):
    """Average-bin the last two axes of a stack by integer factor b."""
    if b == 1:
        return a
    s = a.shape
    return a.reshape(s[0], s[1] // b, b, s[2] // b, b).mean(2).mean(-1)


def paganin(data2d, wavelength, voxelsize, distance, delta_beta, alpha=1e-6):
    """Single-distance Paganin phase retrieval (the standard near-field init)."""
    fx = xp.fft.fftfreq(data2d.shape[-1], d=voxelsize).astype("float32")
    fx, fy = xp.meshgrid(fx, fx)
    num = (1 + wavelength * distance * xp.pi * delta_beta * (fx**2 + fy**2)) \
        * xp.fft.fft2(data2d)
    den = (1 + wavelength * distance * xp.pi * delta_beta * (fx**2 + fy**2))**2 + alpha
    phase = delta_beta * 0.5 * xp.log(xp.real(xp.fft.ifft2(num / den)))
    return phase


def main():
    # --------------------------- load data ------------------------------- #
    with h5py.File(DATA, "r") as f:
        data = np.asarray(f["data"][()], dtype="float32")          # (npos, N, N)
        posx = np.asarray(f["posx_um"][()], dtype="float64")
        posy = np.asarray(f["posy_um"][()], dtype="float64")
        energy = float(f["energy_keV"][()])
        distance = float(f["distance_mm"][()]) * 1e-3              # -> m
        pixelsize = float(f["pixelsize_um"][()]) * 1e-6           # -> m (sample plane)

    wavelength = 1.24e-9 / energy
    # Binning coarsens the detector: voxel and n scale, distance is unchanged.
    data = bin2(data, BIN)
    voxelsize = pixelsize * BIN
    npos, n, _ = data.shape

    # Geometry sizes (mirror the reference demo)
    pad = n // 16
    ex = 8
    npsi = n + n // 4
    nq = n + 2 * pad

    # Positions: micrometres -> sample-plane pixels; pos[:,0]=y, pos[:,1]=x
    pos = np.stack([posy * 1e-6 / voxelsize,
                    posx * 1e-6 / voxelsize], axis=1).astype("float32")
    pos -= pos.mean(0)  # center the position cloud

    print("energy=%.2f keV  lambda=%.3e m  distance=%.4f m  voxel=%.1f nm (bin %d)"
          % (energy, wavelength, distance, voxelsize * 1e9, BIN), flush=True)
    print("npos=%d  n=%d  npsi=%d  nq=%d  pos span px: y=%.0f x=%.0f"
          % (npos, n, npsi, nq, np.ptp(pos[:, 0]), np.ptp(pos[:, 1])), flush=True)

    data = xp.asarray(data)
    pos = xp.asarray(pos)

    bh = BHPtycho(n, npsi, pad, ex, npos, voxelsize, distance, wavelength, xp=xp)

    # ----------------------- initial guesses ----------------------------- #
    print("building initial guess (probe + Paganin object, delta_beta=%.0f)..."
          % DELTA_BETA, flush=True)
    ri = xp.round(pos).astype("int32")
    # reference (empty-beam proxy): average data over positions
    dref = data.mean(0)
    q_init = bh.DT(xp.sqrt(dref[None]))[0].astype("complex64")

    # object: Paganin on each normalised frame, stitched into the object grid
    rdata = data / (dref[None] + 1e-9)
    obj_phase = xp.zeros((npsi, npsi), dtype="float32")
    weight = xp.zeros((npsi, npsi), dtype="float32")
    sx = bh._to_host(npsi // 2 - ri[:, 1] - n // 2)
    sy = bh._to_host(npsi // 2 - ri[:, 0] - n // 2)
    for k in range(npos):
        ph = paganin(rdata[k], wavelength, voxelsize, distance, DELTA_BETA)
        obj_phase[sy[k]:sy[k] + n, sx[k]:sx[k] + n] += ph
        weight[sy[k]:sy[k] + n, sx[k]:sx[k] + n] += 1
    weight[weight < 0.5] = 1
    psi_init = xp.exp(1j * (obj_phase / weight)).astype("complex64")

    # ----------------------- reconstruction ------------------------------ #
    print("starting BH-CG (%d iters). First GPU iteration includes cuFFT "
          "planning and can take a while; later iterations are fast." % NITER,
          flush=True)
    t0 = time.time()
    errs = []

    def cb(i, err, v):
        # print every iteration so progress is always visible
        print("  iter %3d   error = %.5e   (%.1fs)" % (i, err, time.time() - t0),
              flush=True)

    psi, q, errs = bh.reconstruct(data, psi_init.copy(), q_init.copy(), pos,
                                  niter=NITER, method="BH-CG", rho=RHO, callback=cb)
    print("done: error %.4e -> %.4e (%.2fx) in %d iters, %.1fs"
          % (errs[0], errs[-1], errs[-1] / errs[0], NITER, time.time() - t0),
          flush=True)

    # ----------------------- save + plot --------------------------------- #
    to_host = (lambda a: a.get()) if USE_GPU else (lambda a: a)
    out = "recons/bh"
    os.makedirs(out, exist_ok=True)
    np.savez(out + "/bh_result.npz",
             psi=to_host(psi), q=to_host(q), errors=np.asarray(errs))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ph = np.angle(to_host(psi))
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        im0 = ax[0].imshow(ph, cmap="gray"); ax[0].set_title("object phase"); plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(np.abs(to_host(q)), cmap="gray"); ax[1].set_title("probe amplitude"); plt.colorbar(im1, ax=ax[1])
        ax[2].semilogy(errs); ax[2].set_title("BH-CG convergence"); ax[2].set_xlabel("iter"); ax[2].grid(True)
        fig.tight_layout(); fig.savefig(out + "/bh_result.png", dpi=130)
        print("saved", out + "/bh_result.png and .npz")
    except Exception as e:
        print("plot skipped (%s); data saved to .npz" % e)


if __name__ == "__main__":
    main()
