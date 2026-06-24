# -*- coding: utf-8 -*-
"""
Bilinear-Hessian (BH) near-field ptychography -- a clean, portable, documented
reimplementation of the second-order method of

    Carlsson, Wendt, Cloetens & Nikitin,
    "Efficient near-field ptychography reconstruction using the Hessian
    operator", Opt. Express 33, 30543 (2025),

distilled from V. Nikitin's reference code (github.com/nikitinvv/BH-ptychography).

WHY THIS METHOD
---------------
The slow low-frequency / halo convergence of Difference Map is a *conditioning*
problem: the near-field objective's Hessian is very ill-conditioned, with the
low-frequency (DC-ish) object modes at the tiny-eigenvalue end (the Fresnel CTF
barely responds to them). Any first-order method converges at a rate set by that
condition number, so those modes crawl -- hence thousands of iterations.

A second-order method rescales each mode by its own curvature, so low- and
high-frequency modes converge at comparable rates. This implementation does that
*without ever forming the Hessian matrix*: it only needs Hessian-vector products,
built from the same FFTs as the forward model. The result is conjugate-gradient
with an analytically optimal (Newton) step length and conjugation coefficient.

THE WHOLE ALGORITHM, IN WORDS
-----------------------------
forward:     big_psi = D( S(psi)*q )           # shift+crop object, x probe, propagate
objective:   f = || |big_psi| - sqrt(I) ||^2    # amplitude (Gaussian) model
residual:    big_phi = D^T( 2(big_psi - d*big_psi/|big_psi|) )
gradients:   grad_psi = S^T(conj(q)*big_phi),  grad_q = sum_k conj(patch)*big_phi
curvature:   hessian_F(...) = curvature of f in the detector field
step:        alpha = -<grad, eta> / hessian_along_eta     # exact Newton step
direction:   conjugate-gradient eta, with Hessian-based beta
precond:     object/probe rescaled by rho (Appendix II) to balance their scales

PORTABILITY
-----------
Set the array module with `xp=numpy` (CPU, for learning / small tests on a laptop)
or `xp=cupy` (GPU, for production on the cluster). The code is identical; only the
backend changes. Patch extraction is done with plain indexing (fine for the
handful of positions in full-field near-field) instead of a custom CUDA kernel.

authors: distilled & documented for J. C. da Silva
"""
import numpy as np


def _is_smooth(m, primes=(2, 3, 5, 7)):
    """True if m factorizes into only the given small primes (cuFFT-fast)."""
    for p in primes:
        while m % p == 0:
            m //= p
    return m == 1


def _good_fft_size(m, parity=None):
    """Smallest 7-smooth integer >= m (cuFFT-friendly). If `parity` is given
    (0 even, 1 odd), the result also matches that parity -- used so that
    (npatch - nq) stays even and the shift border `ex` is an integer."""
    k = int(m)
    while True:
        if (parity is None or k % 2 == parity) and _is_smooth(k):
            return k
        k += 1


class BHPtycho:
    """
    Geometry + operators + BH-CG reconstruction for near-field ptychography.

    Sizes (all in pixels), mirroring the reference code:
      n      : detector size (per axis)
      npsi   : object size  (> n, holds the full field of view + position spread)
      pad    : padding so the reconstructed probe is a bit larger than the detector
      nq     : probe size = n + 2*pad
      ex     : extra border for sub-pixel shifts
      npatch : patch size = nq + 2*ex
      npos   : number of scan positions
    """

    def __init__(self, n, npsi, pad, ex, npos,
                 voxelsize, distance, wavelength, eps=1e-8, xp=np):
        self.xp = xp
        self.n = n
        self.npsi = npsi
        self.pad = pad
        self.npos = npos
        self.nq = n + 2 * pad
        # The patch FFT length (npatch) must be cuFFT-friendly: a size with a
        # large prime factor (e.g. nq+16 = 16*73) sends cuFFT to the slow
        # Bluestein path and can look like a hang on GPU. Round npatch up to the
        # nearest 7-smooth size by enlarging the shift border `ex` (>= requested).
        self.npatch = _good_fft_size(self.nq + 2 * ex, parity=self.nq % 2)
        self.ex = (self.npatch - self.nq) // 2
        self.voxelsize = voxelsize
        self.distance = distance
        self.wavelength = wavelength
        self.eps = eps

        # Propagator FFT length: pad to >= 2*nq (no wrap-around), 7-smooth.
        self.nprop = _good_fft_size(2 * self.nq, parity=0)
        self._plo = (self.nprop - self.nq) // 2
        self._phi = self.nprop - self.nq - self._plo

        # Near-field Fresnel propagation kernel (angular spectrum) on the nprop grid.
        fx = xp.fft.fftfreq(self.nprop, d=voxelsize).astype("float32")
        fx, fy = xp.meshgrid(fx, fx)
        self.fker = xp.exp(-1j * xp.pi * wavelength * distance * (fx ** 2 + fy ** 2))

    # ------------------------------------------------------------------ #
    # Patch extraction E and its adjoint E^T (scatter-add)               #
    # ------------------------------------------------------------------ #
    def E(self, psi, ri):
        """Extract npos patches of size npatch from the object at integer
        positions ri. Plain indexing -- fine for the few near-field positions."""
        xp = self.xp
        npatch, npsi = self.npatch, self.npsi
        stx = npsi // 2 - ri[:, 1] - npatch // 2
        sty = npsi // 2 - ri[:, 0] - npatch // 2
        res = xp.empty([len(stx), npatch, npatch], dtype="complex64")
        sx = self._to_host(stx)
        sy = self._to_host(sty)
        for k in range(len(sx)):
            res[k] = psi[sy[k]:sy[k] + npatch, sx[k]:sx[k] + npatch]
        return res

    def ET(self, psi, psir, ri):
        """Adjoint of E: scatter-add the patches back into the object array."""
        npatch, npsi = self.npatch, self.npsi
        stx = npsi // 2 - ri[:, 1] - npatch // 2
        sty = npsi // 2 - ri[:, 0] - npatch // 2
        sx = self._to_host(stx)
        sy = self._to_host(sty)
        for k in range(len(sx)):
            psi[sy[k]:sy[k] + npatch, sx[k]:sx[k] + npatch] += psir[k]
        return psi

    def _to_host(self, a):
        """Get a small index array as host ints (works for numpy or cupy)."""
        return (a.get() if hasattr(a, "get") else np.asarray(a)).astype(int)

    # ------------------------------------------------------------------ #
    # Near-field Fresnel propagator D and its adjoint D^T               #
    # ------------------------------------------------------------------ #
    def D(self, psi):
        """Forward near-field propagation, then crop probe->detector size."""
        xp = self.xp
        nq, pad, plo = self.nq, self.pad, self._plo
        ff = xp.pad(psi, ((0, 0), (plo, self._phi), (plo, self._phi)))  # nq -> nprop
        ff = xp.fft.ifft2(xp.fft.fft2(ff) * self.fker)
        ff = ff[:, plo:plo + nq, plo:plo + nq]                          # -> nq (center)
        ff = ff[:, pad:nq - pad, pad:nq - pad]                          # -> n (detector)
        return ff

    def DT(self, psi):
        """Adjoint propagation: pad detector->probe, back-propagate."""
        xp = self.xp
        nq, pad, plo = self.nq, self.pad, self._plo
        ff = xp.pad(psi, ((0, 0), (pad, pad), (pad, pad)))              # n -> nq
        ff = xp.pad(ff, ((0, 0), (plo, self._phi), (plo, self._phi)))   # nq -> nprop
        ff = xp.fft.ifft2(xp.fft.fft2(ff) / self.fker)
        ff = ff[:, plo:plo + nq, plo:plo + nq]                          # -> nq (center)
        return ff

    # ------------------------------------------------------------------ #
    # Sub-pixel shifted extraction S and its adjoint S^T                #
    # ------------------------------------------------------------------ #
    def _shift_phasor(self, r, sign):
        xp = self.xp
        npatch = self.npatch
        x = xp.fft.fftfreq(npatch).astype("float32")
        y, x = xp.meshgrid(x, x)
        return xp.exp(sign * 2 * xp.pi * 1j *
                      (y * r[:, 1, None, None] + x * r[:, 0, None, None])
                      ).astype("complex64")

    def S(self, psi, ri, r):
        """Extract patches with integer (ri) + sub-pixel (r) shift; crop the ex
        border so the returned patch is probe-sized (nq)."""
        xp = self.xp
        ex, nq = self.ex, self.nq
        psir = self.E(psi, ri)
        psir = xp.fft.ifft2(self._shift_phasor(r, -1) * xp.fft.fft2(psir))
        return psir[:, ex:self.npatch - ex, ex:self.npatch - ex]

    def ST(self, d, ri, r):
        """Adjoint of S: pad ex border, inverse sub-pixel shift, scatter-add."""
        xp = self.xp
        ex, npsi = self.ex, self.npsi
        psi = xp.zeros([npsi, npsi], dtype="complex64")
        psir = xp.pad(d, ((0, 0), (ex, ex), (ex, ex)))
        psir = xp.fft.ifft2(self._shift_phasor(r, +1) * xp.fft.fft2(psir))
        self.ET(psi, psir, ri)
        return psi

    # ------------------------------------------------------------------ #
    # Forward model, objective, residual                                 #
    # ------------------------------------------------------------------ #
    def forward(self, psi, q, ri, r):
        """big_psi = D( S(psi)*q ): the predicted complex field at the detector."""
        return self.D(self.S(psi, ri, r) * q)

    def minF(self, big_psi, d):
        """Amplitude (Gaussian) data-fit objective: || |big_psi| - d ||^2."""
        return float(self.xp.linalg.norm(self.xp.abs(big_psi) - d) ** 2)

    def _big_phi(self, big_psi, d):
        """Back-propagated amplitude residual used to assemble the gradients.
        td = d * big_psi/|big_psi| replaces the modulus by the measured one
        while keeping the model phase (the amplitude projection)."""
        xp = self.xp
        td = d * (big_psi / (xp.abs(big_psi) + self.eps))
        return self.DT(2 * (big_psi - td))

    # ------------------------------------------------------------------ #
    # Gradients                                                          #
    # ------------------------------------------------------------------ #
    def gradient_psi(self, q, ri, r, big_phi):
        return self.ST(self.xp.conj(q) * big_phi, ri, r)

    def gradient_q(self, spsi, big_phi):
        return self.xp.sum(self.xp.conj(spsi) * big_phi, axis=0)

    # ------------------------------------------------------------------ #
    # The bilinear Hessian of the amplitude objective (detector field)   #
    # ------------------------------------------------------------------ #
    def hessian_F(self, big_psi, dbig1, dbig2, d):
        """Curvature of f = || |big_psi| - d ||^2 in the detector field, applied
        to two perturbation directions dbig1, dbig2. Two terms: a (1 - d/|Psi|)
        amplitude-curvature term and a d/|Psi| phasor-projected term."""
        xp = self.xp
        l0 = big_psi / (xp.abs(big_psi) + self.eps)
        d0 = d / (xp.abs(big_psi) + self.eps)
        v1 = xp.sum((1 - d0) * self._reprod(dbig1, dbig2))
        v2 = xp.sum(d0 * self._reprod(l0, dbig1) * self._reprod(l0, dbig2))
        return 2 * (v1 + v2)

    @staticmethod
    def _reprod(a, b):
        """Elementwise real part of a * conj(b) (real inner-product density)."""
        return a.real * b.real + a.imag * b.imag

    def _redot(self, a, b):
        """Real inner product summed over all elements."""
        return float(self.xp.sum(self._reprod(a, b)))

    # ------------------------------------------------------------------ #
    # Newton-optimal step length and CG conjugation coefficient          #
    # ------------------------------------------------------------------ #
    def _shifted_dir(self, dpsi, ri, r):
        """S applied to an object-direction without sub-pixel re-crop subtlety:
        returns the probe-sized shifted patch of dpsi (used inside the Hessian)."""
        xp = self.xp
        ex, nq = self.ex, self.nq
        tmp = xp.fft.fft2(self.E(dpsi, ri))
        return xp.fft.ifft2(self._shift_phasor(r, -1) * tmp)[:, ex:nq + ex, ex:nq + ex]

    def calc_alpha(self, vars_, grads, etas, reused, d):
        """Exact step length alpha = -<grad, eta> / Hessian[eta, eta].
        This is the line minimizer of the 2nd-order Taylor model along eta --
        the single most important ingredient for fast convergence."""
        q, ri, r = vars_["q"], vars_["ri"], vars_["r"]
        dpsi1, dq1 = grads["psi"], grads["q"]
        dpsi2, dq2 = etas["psi"], etas["q"]
        spsi, big_psi, big_phi = reused["spsi"], reused["big_psi"], reused["big_phi"]

        top = -self._redot(dpsi1, dpsi2) - self._redot(dq1, dq2)

        sdpsi = self._shifted_dir(dpsi2, ri, r)
        d2m2 = 2 * dq2 * sdpsi               # second-order (probe x object) term
        dm = dq2 * spsi + q * sdpsi          # first-order forward perturbation
        Ddm = self.D(dm)
        bottom = self._redot(big_phi, d2m2) + self.hessian_F(big_psi, Ddm, Ddm, d)
        return top / bottom, top, bottom

    def calc_beta(self, vars_, grads, etas, reused, d, rho_sq):
        """Hessian-based conjugate-gradient coefficient (Daniel's rule)."""
        q, ri, r = vars_["q"], vars_["ri"], vars_["r"]
        spsi, big_psi, big_phi = reused["spsi"], reused["big_psi"], reused["big_phi"]

        dpsi1, dq1 = grads["psi"] * rho_sq[0], grads["q"] * rho_sq[1]
        dpsi2, dq2 = etas["psi"], etas["q"]

        sdpsi1 = self._shifted_dir(dpsi1, ri, r)
        sdpsi2 = self._shifted_dir(dpsi2, ri, r)

        d2m1 = dq1 * sdpsi2 + dq2 * sdpsi1
        d2m2 = 2 * dq2 * sdpsi2
        dm1 = dq1 * spsi + q * sdpsi1
        dm2 = dq2 * spsi + q * sdpsi2
        Ddm1, Ddm2 = self.D(dm1), self.D(dm2)

        top = self._redot(big_phi, d2m1) + self.hessian_F(big_psi, Ddm1, Ddm2, d)
        bottom = self._redot(big_phi, d2m2) + self.hessian_F(big_psi, Ddm2, Ddm2, d)
        return top / bottom

    # ------------------------------------------------------------------ #
    # Main BH-CG reconstruction loop                                     #
    # ------------------------------------------------------------------ #
    def reconstruct(self, data, psi, q, pos, niter=100, method="BH-CG",
                    rho=(1.0, 2.0), err_step=1, callback=None):
        """
        Reconstruct object `psi` and probe `q` from near-field intensities `data`
        ([npos, n, n]) at scan positions `pos` ([npos, 2], pixels).

        rho : (rho_psi, rho_q) variable preconditioning scales (Appendix II).
        method : 'BH-CG' (recommended), 'BH-GD' (gradient descent baseline).
        Returns (psi, q, errors).
        """
        xp = self.xp
        d = xp.sqrt(data)
        rho_sq = (xp.asarray(rho).astype("float32")) ** 2

        ri = xp.floor(pos).astype("int32")
        r = (pos - ri).astype("float32")
        vars_ = {"q": q, "psi": psi, "ri": ri, "r": r}

        etas = {}
        errors = []
        for i in range(niter):
            # forward + cached quantities
            spsi = self.S(psi, ri, r)
            big_psi = self.D(spsi * q)
            big_phi = self._big_phi(big_psi, d)
            reused = {"spsi": spsi, "big_psi": big_psi, "big_phi": big_phi}

            if err_step and (i % err_step == 0):
                err = self.minF(big_psi, d)
                errors.append(err)
                if callback is not None:
                    callback(i, err, vars_)

            # gradients
            grads = {"psi": self.gradient_psi(q, ri, r, big_phi),
                     "q": self.gradient_q(spsi, big_phi)}

            # search direction (steepest descent on iter 0, else CG)
            if i == 0 or method == "BH-GD":
                etas["psi"] = -rho_sq[0] * grads["psi"]
                etas["q"] = -rho_sq[1] * grads["q"]
            else:
                beta = self.calc_beta(vars_, grads, etas, reused, d, rho_sq)
                etas["psi"] = -rho_sq[0] * grads["psi"] + beta * etas["psi"]
                etas["q"] = -rho_sq[1] * grads["q"] + beta * etas["q"]

            # Newton-optimal step length and update
            alpha, _, _ = self.calc_alpha(vars_, grads, etas, reused, d)
            psi = psi + alpha * etas["psi"]
            q = q + alpha * etas["q"]
            vars_["psi"], vars_["q"] = psi, q

        return psi, q, errors
