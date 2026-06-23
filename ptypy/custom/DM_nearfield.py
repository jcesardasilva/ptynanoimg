# -*- coding: utf-8 -*-
"""
An extension plugin of the Difference Map engine, tailored to the slow
low-frequency convergence of *near-field* ptychography.

Motivation
----------
In near-field ptychography the propagation is a Fresnel chirp whose contrast
transfer function (CTF) for a weak/pure-phase object behaves like

    CTF(f) ~ sin( pi * lam * z * f**2 )

which vanishes at low spatial frequencies (f -> 0). The measured intensity is
therefore almost blind to the low-frequency phase content of the object. Two
practical consequences follow:

1. Low frequencies relax extremely slowly under Difference Map, producing the
   familiar *halo* artefact around the sample that only disappears after many
   thousands of iterations.
2. The usual error metrics live in the detector plane, where those same low
   frequencies carry near-zero weight (CTF ~ 0). The metric is effectively
   CTF-weighted and so is blind to exactly the modes that converge slowest --
   it cannot "see" the halo decaying.

This engine adds two opt-in tools:

* **An object-plane low-frequency convergence metric** (Part 1) so the halo's
  evolution can actually be monitored. It is recorded in
  ``ptycho.runtime.iter_info[-1]['nf_metric']`` and logged each iteration.

* **A Fourier-domain preconditioner of the object update** (Part 2) that boosts
  the poorly-constrained low frequencies of the *object increment* (not the
  object itself), accelerating the slow modes while leaving the DM fixed point
  unchanged (at convergence the increment -> 0, so the boost does nothing).

Both features are disabled by default; switching either on reproduces plain DM
up to the enabled behaviour.

authors: J. C. da Silva
"""
import numpy as np

from ptypy.engines import projectional
from ptypy.engines import register
from ptypy.utils import parallel
from ptypy.utils.verbose import log, ilog_message


@register()
class DMNearfield(projectional.DM):
    """
    Difference Map with near-field low-frequency acceleration and diagnostics.

    Defaults:

    [nf_metric]
    default = True
    type = bool
    help = Track an object-plane low-frequency convergence metric
    doc = When enabled, the relative iteration-to-iteration change of the\
          low-pass-filtered object is recorded under the 'nf_metric' key of the\
          runtime iteration info and logged. This is sensitive to the halo that\
          the detector-plane error is blind to.

    [nf_metric_cutoff]
    default = 0.1
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Low-pass cutoff for the metric, as a fraction of the Nyquist frequency
    doc = The metric measures the change of the object band below this cutoff.\
          0.1 keeps the lowest 10 percent of frequencies, i.e. the band most\
          affected by the near-field CTF zero.

    [nf_metric_interval]
    default = 1
    type = int
    lowlim = 1
    help = Compute the low-frequency metric every N iterations
    doc = Each evaluation costs two FFTs over the full object, so increase this\
          for very large objects if the diagnostic is not needed every iteration.

    [nf_precond]
    default = False
    type = bool
    help = Enable the Fourier-domain preconditioner of the object update
    doc = Amplifies the low-frequency content of the object *increment* to\
          accelerate the slowly-converging modes. The DM fixed point is\
          preserved because the boost acts on the per-iteration increment only.\
          WARNING: in near-field the low frequencies are only weakly constrained\
          (CTF ~ 0), so an over-aggressive boost is positive feedback and will\
          diverge. Keep nf_precond_gain modest (<= ~2), prefer the 'ctf' method,\
          and use nf_precond_ramp to avoid a sudden jolt.

    [nf_precond_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before preconditioning starts

    [nf_precond_stop]
    default = None
    type = int
    help = Number of iterations after which preconditioning stops (None = never)

    [nf_precond_ramp]
    default = 20
    type = int
    lowlim = 0
    help = Number of iterations over which the boost ramps linearly from 1x to full
    doc = Switching the boost on abruptly kicks the worst-conditioned modes and\
          can destabilise near-field reconstructions. The effective boost grows\
          from identity to the full filter over this many iterations after\
          nf_precond_start. Set to 0 to switch on at full strength immediately.

    [nf_precond_method]
    default = ctf
    type = str
    help = Preconditioner shape; 'ctf' (regularised inverse-CTF from the geometry, slowness-matched) or 'gain' (flat Gaussian low-frequency boost)
    doc = 'ctf' amplifies each mode in proportion to how blind the near-field\
          response is to it, so well-constrained modes are left near 1x. 'gain'\
          boosts the whole low band equally and is more prone to overshoot.
    choices = ['ctf', 'gain']

    [nf_precond_gain]
    default = 2.0
    type = float
    lowlim = 1.0
    help = Maximum amplification applied to the low frequencies of the increment
    doc = Values much above ~2 are unsafe for near-field (positive feedback on\
          weakly-constrained low frequencies). Increase cautiously.

    [nf_precond_cutoff]
    default = 0.1
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Cutoff of the 'gain' boost as a fraction of the Nyquist frequency

    [nf_precond_reg]
    default = 0.1
    type = float
    lowlim = 0.0
    help = Regularisation (relative to max CTF^2) for the 'ctf' preconditioner
    doc = Larger values are more conservative, capping the amplification near\
          the CTF zeros to suppress noise.

    [nf_lf_constraint]
    default = False
    type = bool
    help = Pin the low-frequency phase reference using a known-empty region
    doc = The single-distance near-field CTF is blind at f=0, so the smooth\
          (low-order) phase background is unconstrained and drifts as the halo.\
          When enabled, a low-order 2D polynomial is fitted to the object phase\
          over the `nf_lf_mask` region (which you know should be flat) and\
          subtracted from the whole object each iteration, anchoring that\
          reference. This complements the preconditioner: the preconditioner\
          speeds up the weakly-constrained modes, the constraint pins the\
          truly-unconstrained ones.

    [nf_lf_mask]
    default = None
    type = ndarray
    help = Boolean mask (object Y-X shape), True over a known-empty/flat region
    doc = Manual mask of the flat/vacuum region used to anchor the phase. Cast\
          to bool, must match the last two axes of the object storage. If None\
          and nf_lf_auto is True, the mask is detected from the data instead.

    [nf_lf_auto]
    default = True
    type = bool
    help = Auto-detect the flat/vacuum region from the current object estimate
    doc = Avoids hard-coding a mask. The flattest fraction of the field (low\
          phase/amplitude gradient and amplitude near the modal transmission) is\
          selected as the anchor region and refreshed periodically. A manual\
          nf_lf_mask, when provided, always takes precedence.

    [nf_lf_auto_quantile]
    default = 0.3
    type = float
    lowlim = 0.01
    uplim = 0.9
    help = Fraction of the flattest pixels selected as the auto anchor region
    doc = 0.3 keeps the smoothest 30 percent of the field. Lower is stricter\
          (only the very flattest vacuum); higher includes more of the field.

    [nf_lf_auto_refresh]
    default = 10
    type = int
    lowlim = 1
    help = Recompute the auto mask every N iterations
    doc = The object improves as iterations proceed, so the detected flat region\
          is refreshed periodically rather than fixed once.

    [nf_lf_order]
    default = 2
    type = int
    lowlim = 0
    uplim = 4
    help = Polynomial order of the background phase removed by the constraint
    doc = 0 removes a constant phase offset, 1 also removes tilt (a phase ramp),\
          2 additionally removes the quadratic curvature that dominates the\
          near-field halo (residual parabolic phase from the CTF zero at DC).

    [nf_lf_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before the low-frequency constraint starts

    [nf_lf_pin_amplitude]
    default = False
    type = bool
    help = Also reset the object amplitude in the masked region to unity
    doc = Use when the known-empty region should be pure vacuum (unit amplitude).

    [nf_probe_lf_anchor]
    default = False
    type = bool
    help = Anchor the probe low frequencies to remove the object-probe LF gauge
    doc = The object and probe share a low-frequency gauge freedom (O -> O g,\
          P -> P / g) that in near-field is a near-null direction of the\
          iteration -- the dominant cause of slow low-frequency convergence.\
          When enabled, the low-frequency content of the per-iteration probe\
          *increment* is attenuated, so low-frequency corrections are absorbed\
          by the object rather than drifting into the probe. This anchors the\
          gauge to the (well-characterised) initial probe wavefront while still\
          letting the speckle / high-frequency probe structure refine. It is the\
          generalisation of seeding a good probe and freezing it early.

    [nf_probe_lf_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before the probe low-frequency anchor starts

    [nf_probe_lf_cutoff]
    default = 0.1
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Probe-anchor low-frequency band, as a fraction of the Nyquist frequency

    [nf_probe_lf_damp]
    default = 0.0
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Retained fraction of the low-frequency probe increment (0 = freeze, 1 = off)
    doc = 0.0 fully freezes the probe low frequencies (strongest gauge fix);\
          intermediate values let them adapt slowly. High frequencies always\
          update at full strength.

    [nf_anderson]
    default = False
    type = bool
    help = Anderson-accelerate the DM object fixed-point iteration
    doc = Anderson (DIIS) extrapolation from the last few object iterates cancels\
          the slowest-decaying eigenmodes of the iteration -- which in near-field\
          are the low-frequency modes -- often cutting the iteration count\
          substantially. Wraps whatever inner update is configured (it composes\
          with the preconditioner / constraints). Memory cost: ~2*depth copies of\
          the object. Object-only; the probe follows the plain DM update.

    [nf_anderson_depth]
    default = 4
    type = int
    lowlim = 1
    uplim = 20
    help = Number of past iterates used in the Anderson extrapolation (history m)

    [nf_anderson_start]
    default = 5
    type = int
    lowlim = 0
    help = Number of iterations before Anderson acceleration starts
    doc = Let plain DM establish coarse structure first; extrapolating from a\
          wild early transient can destabilise.

    [nf_anderson_beta]
    default = 1.0
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Anderson mixing (relaxation) factor; 1.0 is the standard choice

    [nf_anderson_reg]
    default = 1e-8
    type = float
    lowlim = 0.0
    help = Relative ridge regularisation for the Anderson least-squares solve
    doc = Stabilises the (often ill-conditioned) extrapolation. Increase if the\
          accelerated iteration becomes noisy.

    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        # Per-storage cache of the previous low-pass object (for the metric)
        self._nf_prev_lowpass = {}
        # Per-storage cache of the preconditioner filter, keyed by (name, shape)
        self._nf_filter_cache = {}
        # Per-storage cache of the auto anchor mask: name -> (iter, mask)
        self._nf_auto_mask = {}
        # Anderson acceleration history: name -> {'x': [...], 'f': [...]}
        self._nf_aa = {}
        # Latest metric value, surfaced through _fill_runtime
        self._nf_metric_value = None

    # ------------------------------------------------------------------ #
    # Part 2: Fourier-domain preconditioner of the object update         #
    # ------------------------------------------------------------------ #
    def _nf_precond_active(self):
        if not self.p.nf_precond:
            return False
        if self.curiter < self.p.nf_precond_start:
            return False
        if (self.p.nf_precond_stop is not None) and (self.curiter >= self.p.nf_precond_stop):
            return False
        return True

    def _nf_geometry_for(self, storage):
        """
        Find a Geometry instance whose object view lives in `storage`.
        Returns None if none can be located (e.g. not yet prepared).
        """
        for pod in self.pods.values():
            if not pod.active:
                continue
            if pod.ob_view.storage is storage:
                return pod.geometry
        return None

    def _nf_ramp_fraction(self):
        """
        Linear ramp of the boost strength from 0 (identity) to 1 (full filter)
        over `nf_precond_ramp` iterations after `nf_precond_start`. Avoids the
        destabilising jolt of switching a strong boost on abruptly.
        """
        ramp = self.p.nf_precond_ramp
        if ramp <= 0:
            return 1.0
        frac = (self.curiter - self.p.nf_precond_start) / float(ramp)
        return float(np.clip(frac, 0.0, 1.0))

    def _nf_build_boost(self, name, storage):
        """
        Build (and cache) the Fourier-domain *boost shape* (filter - 1, so it is
        >= 0 and zero where no amplification is wanted) for a given object
        storage, evaluated on its last two axes. The effective per-iteration
        filter is 1 + frac * boost, where frac is the ramp fraction.
        """
        shape = storage.data.shape[-2:]
        key = (name, shape, self.p.nf_precond_method)
        cached = self._nf_filter_cache.get(key)
        if cached is not None:
            return cached

        ny, nx = shape
        if self.p.nf_precond_method == 'ctf':
            geo = self._nf_geometry_for(storage)
            if geo is None:
                log(2, "DMNearfield: no geometry found for object storage %s; "
                       "falling back to 'gain' preconditioner." % name)
                filt = self._nf_gain_filter(ny, nx)
            else:
                filt = self._nf_ctf_filter(ny, nx, geo)
        else:
            filt = self._nf_gain_filter(ny, nx)

        boost = (filt - 1.0).astype(np.float32)
        self._nf_filter_cache[key] = boost
        return boost

    def _nf_gain_filter(self, ny, nx):
        """
        Smooth low-frequency boost: G(f) = 1 + (gain-1) * exp(-(|f|/fc)^2),
        with f in normalised frequency (Nyquist = 0.5). Monotone, no CTF-zero
        spikes -- the robust default.
        """
        fy = np.fft.fftfreq(ny)[:, None]
        fx = np.fft.fftfreq(nx)[None, :]
        f2 = fy ** 2 + fx ** 2
        fc = self.p.nf_precond_cutoff * 0.5
        gauss = np.exp(-f2 / (fc ** 2 + 1e-12))
        return 1.0 + (self.p.nf_precond_gain - 1.0) * gauss

    def _nf_ctf_filter(self, ny, nx, geo):
        """
        Regularised inverse-CTF boost built from the actual near-field geometry:

            CTF(f) = sin(chi(f)),  chi(f) = 2*pi*(z/lam)*(sqrt(1 - (lam f)^2) - 1)
            G(f)   = clip( (CTF_max^2 + a) / (CTF(f)^2 + a), 1, gain )

        where a = reg * CTF_max^2. This selectively amplifies the bands where
        the near-field response is weak (low frequencies and the CTF zeros),
        capped by `nf_precond_gain` for stability.
        """
        lam = float(geo.lam)
        z = float(geo.p.distance)
        # resolution is the sample-plane pixel size (dy, dx) in metres
        res = np.asarray(geo.resolution, dtype=np.float64).ravel()
        dy, dx = (res[0], res[-1])

        fy = np.fft.fftfreq(ny, d=dy)[:, None]
        fx = np.fft.fftfreq(nx, d=dx)[None, :]
        a2 = (lam * fy) ** 2 + (lam * fx) ** 2
        # Evanescent / out-of-band frequencies: treat as fully blind (CTF=0)
        a2 = np.clip(a2, 0.0, 1.0)
        chi = 2.0 * np.pi * (z / lam) * (np.sqrt(1.0 - a2) - 1.0)
        ctf2 = np.sin(chi) ** 2

        ctf2_max = ctf2.max()
        if ctf2_max <= 0:
            return self._nf_gain_filter(ny, nx)
        reg = self.p.nf_precond_reg * ctf2_max
        gain = (ctf2_max + reg) / (ctf2 + reg)
        return np.clip(gain, 1.0, self.p.nf_precond_gain)

    def object_update(self):
        """
        Standard DM object update, optionally followed by (i) a Fourier-domain
        boost of the low-frequency content of the per-iteration increment and
        (ii) a low-frequency phase-reference constraint.
        """
        self._nf_object_update_core()
        if self._nf_lf_active():
            self._nf_apply_lf_constraint()

    def _nf_object_update_core(self):
        if not self._nf_precond_active():
            super().object_update()
            return

        frac = self._nf_ramp_fraction()
        if frac <= 0.0:
            # Start of the ramp, zero boost strength: plain DM update
            super().object_update()
            return

        # Snapshot the object so we can isolate this iteration's increment
        snapshot = {name: s.data.copy() for name, s in self.ob.storages.items()}

        super().object_update()

        for name, s in self.ob.storages.items():
            boost = self._nf_build_boost(name, s)
            # Effective filter, ramped: 1 + frac * (filter - 1)
            filt = 1.0 + frac * boost
            delta = s.data - snapshot[name]
            # Boost low frequencies of the increment (FFT over last two axes)
            delta_ft = np.fft.fft2(delta, axes=(-2, -1))
            delta_ft *= filt
            boosted = np.fft.ifft2(delta_ft, axes=(-2, -1)).astype(s.data.dtype)
            s.data[:] = snapshot[name] + boosted
            self.clip_object(s)

    # ------------------------------------------------------------------ #
    # Optional: probe low-frequency gauge anchor                         #
    # ------------------------------------------------------------------ #
    def _nf_probe_anchor_active(self):
        return (self.p.nf_probe_lf_anchor
                and self.curiter >= self.p.nf_probe_lf_start)

    def _nf_probe_filter(self, name, storage):
        """
        Attenuation filter for the probe increment: `damp` over the low-frequency
        band, 1 at high frequencies, so D(f) = 1 - (1 - damp) * lowpass(f).
        Cached per (name, shape).
        """
        shape = storage.data.shape[-2:]
        key = ('probe', name, shape)
        cached = self._nf_filter_cache.get(key)
        if cached is not None:
            return cached
        ny, nx = shape
        fy = np.fft.fftfreq(ny)[:, None]
        fx = np.fft.fftfreq(nx)[None, :]
        fc = self.p.nf_probe_lf_cutoff * 0.5
        lowpass = np.exp(-(fy ** 2 + fx ** 2) / (fc ** 2 + 1e-12))
        D = (1.0 - (1.0 - self.p.nf_probe_lf_damp) * lowpass).astype(np.float32)
        self._nf_filter_cache[key] = D
        return D

    def probe_update(self):
        """
        Standard DM probe update, optionally with the low-frequency content of
        the probe increment attenuated to remove the object-probe gauge drift.
        """
        if not self._nf_probe_anchor_active():
            return super().probe_update()

        snapshot = {name: s.data.copy() for name, s in self.pr.storages.items()}
        change = super().probe_update()
        for name, s in self.pr.storages.items():
            D = self._nf_probe_filter(name, s)
            dP = s.data - snapshot[name]
            dP_ft = np.fft.fft2(dP, axes=(-2, -1)) * D
            s.data[:] = snapshot[name] + np.fft.ifft2(
                dP_ft, axes=(-2, -1)).astype(s.data.dtype)
        return change

    # ------------------------------------------------------------------ #
    # Optional: low-frequency phase-reference constraint                 #
    # ------------------------------------------------------------------ #
    def _nf_lf_active(self):
        if not self.p.nf_lf_constraint:
            return False
        if self.p.nf_lf_mask is None and not self.p.nf_lf_auto:
            log(2, "DMNearfield: nf_lf_constraint enabled but nf_lf_mask is "
                   "None and nf_lf_auto is False; skipping the constraint.")
            return False
        return self.curiter >= self.p.nf_lf_start

    def _nf_detect_flat_region(self, ob2d):
        """
        Data-driven detection of the flat/vacuum anchor region from a 2D complex
        object estimate: select the smoothest fraction of the field, i.e. pixels
        with small local gradient AND amplitude close to the modal transmission.
        Returns a boolean mask. No hard-coded geometry.
        """
        a = np.abs(ob2d)
        # Local roughness from the complex gradient (phase + amplitude texture)
        gy, gx = np.gradient(ob2d)
        rough = np.abs(gx) + np.abs(gy)
        # Amplitude deviation from the modal (median) transmission
        amp_dev = np.abs(a - np.median(a))
        # Normalise each cue by its own 95th percentile for scale-invariance
        def _norm(x):
            s = np.percentile(x, 95)
            return x / (s + 1e-12)
        score = _norm(rough) + _norm(amp_dev)
        # Keep the flattest fraction of the field
        thr = np.percentile(score, 100.0 * self.p.nf_lf_auto_quantile)
        mask = score <= thr
        return mask

    def _nf_get_mask(self, name, storage):
        """
        Return the anchor mask for a storage: the manual nf_lf_mask if given,
        otherwise the auto-detected flat region (refreshed every
        nf_lf_auto_refresh iterations and cached per storage).
        """
        if self.p.nf_lf_mask is not None:
            return np.asarray(self.p.nf_lf_mask).astype(bool)
        cache = self._nf_auto_mask
        entry = cache.get(name)
        refresh = (entry is None or
                   (self.curiter - entry[0]) >= self.p.nf_lf_auto_refresh)
        if refresh:
            mask = self._nf_detect_flat_region(storage.data[0])
            cache[name] = (self.curiter, mask)
            log(3, "DMNearfield: auto anchor mask for %s = %d pixels (%.0f%%)"
                   % (name, mask.sum(), 100.0 * mask.mean()))
            return mask
        return entry[1]

    def _nf_poly_basis(self, ny, nx, order):
        """
        Stack of 2D monomial basis images x^i y^j (i+j <= order) on a grid
        normalised to [-1, 1] for conditioning. Returns an (nterms, ny, nx)
        array, cached per (shape, order).
        """
        key = ('polybasis', ny, nx, order)
        cached = self._nf_filter_cache.get(key)
        if cached is not None:
            return cached
        yy, xx = np.meshgrid(np.linspace(-1, 1, ny),
                             np.linspace(-1, 1, nx), indexing='ij')
        terms = []
        for total in range(order + 1):
            for i in range(total + 1):
                terms.append((xx ** i) * (yy ** (total - i)))
        basis = np.asarray(terms, dtype=np.float64)
        self._nf_filter_cache[key] = basis
        return basis

    def _nf_apply_lf_constraint(self):
        """
        Fit a low-order polynomial to the object phase over the flat/vacuum
        anchor region (manual nf_lf_mask or auto-detected) and subtract that
        smooth phase from the whole object, anchoring the otherwise-unconstrained
        low-frequency reference.
        """
        order = int(self.p.nf_lf_order)
        for name, s in self.ob.storages.items():
            ny, nx = s.data.shape[-2:]
            mask = self._nf_get_mask(name, s)
            if mask.shape != (ny, nx):
                log(2, "DMNearfield: anchor mask shape %s != object %s for "
                       "storage %s; skipping." % (mask.shape, (ny, nx), name))
                continue
            basis = self._nf_poly_basis(ny, nx, order)        # (nterms, ny, nx)
            m = mask.ravel()
            A = basis.reshape(basis.shape[0], -1).T            # (npix, nterms)
            Am = A[m]
            # Fit each object mode independently
            for k in range(s.data.shape[0]):
                vals = s.data[k].ravel()[m]
                # Amplitude-weight the fit so flat-vacuum pixels (|.| ~ 1)
                # dominate and near-zero/uncovered pixels contribute nothing.
                w = np.abs(vals)
                # Work relative to the (amplitude-weighted) masked mean to
                # avoid phase wrapping in the fit.
                ref = np.angle(np.sum(vals))
                pm = np.angle(vals * np.exp(-1j * ref))
                sw = np.sqrt(w)
                coeffs, *_ = np.linalg.lstsq(Am * sw[:, None], pm * sw, rcond=None)
                fit_full = (A @ coeffs).reshape(ny, nx) + ref
                s.data[k] *= np.exp(-1j * fit_full).astype(s.data.dtype)
                if self.p.nf_lf_pin_amplitude:
                    # Reset masked amplitude to unity, keep the phase
                    s.data[k][mask] = np.exp(1j * np.angle(s.data[k][mask]))
            self.clip_object(s)

    # ------------------------------------------------------------------ #
    # Part 1: object-plane low-frequency convergence metric              #
    # ------------------------------------------------------------------ #
    def _nf_lowpass(self, data):
        """
        Low-pass the object (complex, over last two axes) with a soft Gaussian
        mask keeping the band below `nf_metric_cutoff` * Nyquist.
        """
        ny, nx = data.shape[-2:]
        fy = np.fft.fftfreq(ny)[:, None]
        fx = np.fft.fftfreq(nx)[None, :]
        f2 = fy ** 2 + fx ** 2
        fc = self.p.nf_metric_cutoff * 0.5
        mask = np.exp(-f2 / (fc ** 2 + 1e-12))
        ft = np.fft.fft2(data, axes=(-2, -1)) * mask
        return np.fft.ifft2(ft, axes=(-2, -1))

    def _nf_compute_metric(self):
        """
        Relative change of the low-pass-filtered object between iterations,
        averaged over object storages. A value that keeps decaying means the
        halo is still evolving; a plateau means the low frequencies have settled.
        Returns None on the very first call (no previous state yet).
        """
        rels = []
        for name, s in self.ob.storages.items():
            lp = self._nf_lowpass(s.data)
            prev = self._nf_prev_lowpass.get(name)
            self._nf_prev_lowpass[name] = lp
            if prev is None or prev.shape != lp.shape:
                continue
            num = np.linalg.norm((lp - prev).ravel())
            den = np.linalg.norm(lp.ravel()) + 1e-12
            rels.append(num / den)
        if not rels:
            return None
        return float(np.mean(rels))

    # ------------------------------------------------------------------ #
    # Optional: Anderson acceleration of the object fixed-point iteration #
    # ------------------------------------------------------------------ #
    def _nf_anderson_active(self):
        return self.p.nf_anderson and self.curiter >= self.p.nf_anderson_start

    def _nf_anderson_apply(self, x_pre):
        """
        Anderson (DIIS) extrapolation on each object storage. `x_pre` holds the
        flattened object iterate x_k from *before* the DM step; the current
        object is the DM image g_k. Produces x_{k+1} by cancelling the slowest
        eigenmodes from the last `depth` iterates and writes it back.
        """
        m = int(self.p.nf_anderson_depth)
        beta = float(self.p.nf_anderson_beta)
        for name, s in self.ob.storages.items():
            g = s.data.ravel().copy()
            x = x_pre[name]
            f = g - x
            hist = self._nf_aa.setdefault(name, {'x': [], 'f': []})
            hist['x'].append(x)
            hist['f'].append(f)
            if len(hist['x']) > m + 1:
                hist['x'].pop(0)
                hist['f'].pop(0)

            n = len(hist['x'])
            if n < 2:
                x_next = x + beta * f
            else:
                Xh, Fh = hist['x'], hist['f']
                dX = np.stack([Xh[i + 1] - Xh[i] for i in range(n - 1)], axis=1)
                dF = np.stack([Fh[i + 1] - Fh[i] for i in range(n - 1)], axis=1)
                # Ridge-regularised normal equations: small (n-1)x(n-1) solve.
                A = dF.conj().T @ dF
                lam = self.p.nf_anderson_reg * (np.trace(A).real / A.shape[0]
                                                + 1e-30)
                A = A + lam * np.eye(A.shape[0], dtype=A.dtype)
                gamma = np.linalg.solve(A, dF.conj().T @ f)
                x_next = x + beta * f - (dX + beta * dF) @ gamma

            if not np.all(np.isfinite(x_next)):
                x_next = g  # fall back to the plain DM step
                hist['x'].clear()
                hist['f'].clear()
            s.data[:] = x_next.reshape(s.data.shape)
            self.clip_object(s)

    def engine_iterate(self, num=1):
        if self._nf_anderson_active():
            x_pre = {name: s.data.ravel().copy()
                     for name, s in self.ob.storages.items()}
            error_dct = super().engine_iterate(num)
            self._nf_anderson_apply(x_pre)
        else:
            error_dct = super().engine_iterate(num)
        if self.p.nf_metric and (self.curiter % self.p.nf_metric_interval == 0):
            self._nf_metric_value = self._nf_compute_metric()
            if self._nf_metric_value is not None:
                # log at INFO (verbose>=3) for the full log, and also emit the
                # always-visible streaming line so it shows at any verbose level
                msg = ("DMNearfield: low-frequency (halo) metric = %.3e"
                       % self._nf_metric_value)
                log(3, msg)
                if parallel.master:
                    ilog_message(msg)
        return error_dct

    def _fill_runtime(self):
        super()._fill_runtime()
        # Attach the metric to the just-appended iteration info
        if self.p.nf_metric and self.ptycho.runtime.iter_info:
            self.ptycho.runtime.iter_info[-1]['nf_metric'] = self._nf_metric_value
