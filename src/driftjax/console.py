"""driftjax.console — human-friendly terminal output for runs.

Design (approved 2026-08): stdlib-only, framed reports + live sweep/solver
progress.  Rules:

* No third-party dependencies.  ANSI colour only when stdout is a TTY and
  ``NO_COLOR`` is unset, so piped/redirected output stays plain+greppable.
* The library stays silent by default: ``simulate(progress=...)`` opts in
  (``None`` = silent, ``True``/"live" = TTY live bar, or your own object
  with ``update(i, **info)`` / callable ``fn(i, info)``).
* Pure reporting — these helpers never touch numerics.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "summary",
    "SweepProgress",
    "machine",
    "hr",
    "thin",
    "section",
    "accent",
    "ok",
    "warn",
    "err",
    "dim",
    "bold",
    "fmt_jsc",
    "fmt_time",
    "TTY",
    "COLOR",
    "WIDTH",
]

WIDTH = 68
# Evaluated at import for backwards compat, but SweepProgress re-evaluates
# dynamically (isatty stale under pytest capsys, see audit P2).
TTY = sys.stdout.isatty()
COLOR = TTY and os.environ.get("NO_COLOR") is None


def _is_tty_now() -> bool:
    try:
        return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    except Exception:
        return False


def _c(s, code):
    # Use dynamic check so color follows current stdout (pytest capture vs TTY)
    try:
        use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    except Exception:
        use_color = False
    return f"\x1b[{code}m{s}\x1b[0m" if use_color else s


def accent(s):
    return _c(s, "1;36")  # cyan bold — headings / numbers


def ok(s):
    return _c(s, "1;32")  # green    — PASS / converged


def warn(s):
    return _c(s, "1;33")  # yellow   — warnings


def err(s):
    return _c(s, "1;31")  # red      — FAILED


def dim(s):
    return _c(s, "2")  # faint    — annotations


def bold(s):
    return _c(s, "1")


def hr(ch="=", width=WIDTH, file=None):
    print(ch * width, file=file or sys.stdout)


def thin(ch="-", width=WIDTH, file=None):
    print(ch * width, file=file or sys.stdout)


def section(title, width=WIDTH, file=None):
    n = max(4, width - len(title) - 5)
    print(f"-- {accent(title)} " + "-" * n, file=file or sys.stdout)


def fmt_jsc(jsc_density):  # A/cm² -> " 20.25 mA/cm²"
    return f"{float(jsc_density) * 1e3:8.2f} mA/cm\xb2"


def fmt_time(s):
    if s is None:
        return "-"
    if s < 60:
        return f"{s:.2f} s"
    return f"{s // 60:.0f} m {s % 60:04.1f} s"


def _g(res, key, default=None):
    """Read a value from a public result object or internal mapping."""
    aliases = {"jsc_density": "jsc", "voc_v": "voc", "eff": "efficiency"}
    attribute = aliases.get(key, key)
    try:
        return getattr(res, attribute)
    except Exception:
        pass
    try:
        return res.get(key, default) if isinstance(res, dict) else getattr(res, key, default)
    except Exception:
        return default


def _iv(res):
    """Return voltage and current arrays from a public result or test mapping."""
    try:
        return res.voltages, res.current
    except AttributeError:
        return res["iv"]


def _layers(des):
    """Compact per-layer description from a design's unique (Chi, Eg) pairs,
    converted from dimensionless to eV via the room-T thermal scale."""
    import numpy as np

    from driftjax.units import thermal_scales

    try:
        energy = thermal_scales(300.0)["energy"]  # eV per dimensionless unit
        chi = np.asarray(des.Chi) * energy
        eg = np.asarray(des.Eg) * energy
        chi = np.round(chi, 2)
        eg = np.round(eg, 2)
    except Exception:
        return ""
    pairs, counts = np.unique(np.stack([chi, eg], 1), axis=0, return_counts=True)
    parts = []
    for (c, e), n in zip(pairs, counts, strict=False):
        parts.append(f"{n}x(Chi {c:g}, Eg {e:g})")
    return " + ".join(parts)


def _thickness_um(des):
    try:
        import numpy as np

        from driftjax.units import thermal_scales

        sc = thermal_scales(300.0)
        return float(np.sum(des.dgrid)) * sc["length"] * 1e4
    except Exception:
        return None


def summary(
    res,
    *,
    title=None,
    des=None,
    runtime=None,
    vmax=None,
    n_steps=None,
    alpha_mode=None,
    tol=None,
    file=None,
):
    """Framed run report for a ``simulate`` result (or any dict/object with
    jsc_density, voc_v, ff, eff; iv for the MPP line).  TTY-safe, plain when
    piped."""
    out = file or sys.stdout
    try:
        import driftjax

        ver = driftjax.__version__
    except Exception:
        ver = "0.1.11"

    jsc = _g(res, "jsc_density")
    voc = _g(res, "voc_v")
    ff = _g(res, "ff")
    eff = _g(res, "eff")

    hr(file=out)
    print(accent(f" driftjax {ver}"), "\u00b7", bold(title or "simulate"), file=out)
    thin(file=out)

    if des is not None:
        layers = _layers(des)
        npts = None
        th = None
        try:
            npts = int(getattr(des, "x", None).size)
        except Exception:
            pass
        th = _thickness_um(des)
        _npts = str(npts) if npts is not None else "\u2014"
        print(f" device   {layers[:30]:30s}  n_points  {_npts}", file=out)
        if th is not None:
            print(f" {'':9s} {'':30s}  thickness {th:.3f} \u00b5m", file=out)
    T = None
    try:
        T = float(_g(res, "cell").T)
    except Exception:
        try:
            T = float(des.T)
        except Exception:
            pass
    alpha = alpha_mode
    if alpha is None:
        try:
            cell = _g(res, "cell")
            alpha = getattr(cell, "alpha_mode", None)
        except Exception:
            alpha = None
    print(
        f" sweep     0 -> {vmax if vmax is not None else '?'} V \u00b7 "
        f"{n_steps if n_steps is not None else '?'} steps" + (f" \u00b7 tol {tol}" if tol else ""),
        file=out,
    )
    print(f" optics    alpha_mode = {alpha or '?':12s}" + (f"  T = {T:g} K" if T else ""), file=out)
    thin(file=out)

    def _finite(x):
        try:
            return float(x), float(x) == float(x) and abs(float(x)) != float("inf")
        except Exception:
            return None, False

    vocf, voc_ok = _finite(voc)
    fff, ff_ok = _finite(ff)
    if jsc is not None and voc is not None:
        if voc_ok:
            print(f" Jsc      {fmt_jsc(jsc)}{'':4s}   Voc   {accent(f'{vocf:.4f} V')}", file=out)
        else:
            bound = f"{float(vmax):.3f} V" if vmax is not None else "sweep max"
            print(
                f" Jsc      {fmt_jsc(jsc)}{'':4s}   Voc   {accent('> ' + bound)}"
                + dim("  (sweep below Voc)"),
                file=out,
            )
    if ff is not None and eff is not None:
        if ff_ok:
            print(
                f" FF       {fff * 100:6.2f} %         \u03b7     {accent(f'{float(eff) * 100:.3f} %')}",
                file=out,
            )
        else:
            print(
                f" FF       {'n/a':>6s}         \u03b7     {accent(f'{float(eff) * 100:.3f} %')}",
                file=out,
            )
    # MPP from the IV: mW/cm² = max(v[V] * j[A/cm²]) * 1e3
    try:
        import numpy as np

        vv, jj = _iv(res)
        vv = np.asarray(vv, dtype=float)
        jj = np.asarray(jj, dtype=float)
        p = vv * jj * 1e3
        imax = int(np.argmax(p))
        print(f" MPP      {p[imax]:8.2f} mW/cm\xb2 @ {vv[imax]:.3f} V", file=out)
    except Exception:
        pass
    thin(file=out)
    if runtime is not None:
        print(
            f" runtime  {fmt_time(runtime)}" + dim("   (first bias step incl. JIT compile)"),
            file=out,
        )
    hr(file=out)


def _num(x, default=0.0):
    """float if finite else None (keeps JSON strict)."""
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except Exception:
        return default


def machine(res, des=None, runtime=None, **extra):
    """Machine-readable summary (for ``--json``)."""
    import numpy as np

    out = {
        "jsc_density_A_per_cm2": _num(_g(res, "jsc_density"), 0.0),
        "voc_V": _num(_g(res, "voc_v"), None),
        "ff": _num(_g(res, "ff"), None),
        "eff_pct": _num(_g(res, "eff"), None),
        "runtime_s": runtime,
    }
    if out["eff_pct"] is not None:
        out["eff_pct"] = out["eff_pct"] * 100.0
    try:
        out["iv"] = {
            "v_V": [float(x) for x in np.asarray(_iv(res)[0])],
            "j_A_per_cm2": [float(x) for x in np.asarray(_iv(res)[1])],
        }
    except Exception:
        pass
    if des is not None:
        try:
            out["design"] = {"n_points": int(getattr(des, "x", None).size)}
        except Exception:
            pass
    out.update(extra)
    return out


class SweepProgress:
    """Live per-bias progress with solver info (iters, residual, backend).

    * TTY            -> a single updating line  ``IV sweep [####··] 17/41 …``
    * non-TTY + verb -> one line per step (log-friendly)
    * non-TTY, quiet -> nothing until ``close()``
    """

    def __init__(
        self,
        n_steps,
        *,
        vmax_v=None,
        label="IV sweep",
        verbose=None,
        file=None,
        design=None,
        alpha_mode=None,
        tol=None,
    ):
        self.n = int(n_steps)
        self.vmax_v = vmax_v
        self.label = label
        self.verbose = verbose if verbose is not None else ("--verbose" in sys.argv)
        self.live = _is_tty_now() and not self.verbose
        self.file = file or sys.stdout
        self.rows = []
        self._closed = False
        self._last_len = 0
        # context for the end-of-run summary (set by simulate)
        self.design = design
        self.alpha_mode = alpha_mode
        self.tol = tol
        self._it = None
        # one-line header: what is being run (dim, single line, no box)
        try:
            n_pts = int(design.x.size) if design is not None else None
        except Exception:
            n_pts = None
        head = f"{label}: {self.n} points"
        if n_pts:
            head += f" · N={n_pts}"
        th = _thickness_um(design) if design is not None else None
        if th:
            head += f" · {th:.2f} um"
        print(dim(head), file=self.file) if self.live or self.verbose else None

    # -- reporting API (called by driftjax.solvers.continuation.sweep) ------
    def update(
        self,
        i,
        *,
        v_v=None,
        iters=None,
        resid=None,
        backend=None,
        fallback=None,
        converged=True,
        dt=None,
        compiling=False,
    ):
        self.rows.append((i, v_v, iters, resid, dt, converged, fallback))
        if compiling:
            if self.live:
                self._draw(f"{self.label}  compiling (first JIT pass)…")
            return
        if self.live:
            frac = (i + 1) / self.n
            bar_w = 22
            filled = int(round(frac * bar_w))
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            cells = []
            if v_v is not None:
                cells.append(f"V = {v_v:.3f} V")
            if iters is not None:
                cells.append(f"{iters} iters")
            if resid is not None:
                cells.append(f"\u2502R\u2502 = {resid:6.1e}")
            if dt is not None:
                cells.append(f"{dt:.2f} s")
            tail = ""
            if not converged:
                tail = err(" FAILED")
            elif fallback:
                tail = dim(f" ({fallback})")
            self._draw(
                f"{self.label}  [{bar}] {i + 1:>{len(str(self.n))}}/{self.n}"
                + "  "
                + "  ".join(cells)
                + tail
            )
        elif self.verbose:
            conv = ok("converged") if converged else err("FAILED")
            fb = f" ({fallback})" if fallback else ""
            print(
                f"  [{i + 1:>{len(str(self.n))}}/{self.n}] "
                f"V = {v_v:.3f} V · {iters} iters · "
                f"\u2502R\u2502 = {resid:.2e} · {dt:.2f} s · {conv}{fb}",
                file=self.file,
            )

    def iter(self, it, err=None, resid=None, phase=None):
        """Per-Newton-iteration hook (wired by simulate/continuation).

        Live mode: ticks the iteration counter in the bar line.
        Verbose mode: one compact line per iteration.
        """
        self._it = it
        label = f"{phase} it={it}" if phase else f"it={it}"
        if self.live:
            self._draw(f"{self.label}  working…  {label}")
        elif self.verbose:
            print(
                f"  [{self.label}] {label}" + (f"  |R| = {resid:.2e}" if resid is not None else ""),
                file=self.file,
            )

    def close(self, res=None):
        if self._closed:
            return
        self._closed = True
        if self.live:
            self.file.write("\r" + " " * self._last_len + "\r")
            self.file.flush()
        if self.verbose and self.rows:
            _absR = "\u2502R\u2502"
            print(
                f"  {'step':>5} {'bias/V':>8} {'iters':>6} {_absR:>10} {'t/s':>7}  status",
                file=self.file,
            )
            for i, v_v, iters, resid, dt, conv, fb in self.rows:
                st = "converged" if conv else (fb or "FAILED")
                print(
                    f"  {i + 1:>5} {v_v:8.3f} {iters:>6} {resid:10.2e} {dt:7.2f}  {st}",
                    file=self.file,
                )
        if res is not None:
            from driftjax.console import summary as _summary

            print(file=self.file)
            _summary(
                res,
                title=self.label,
                des=self.design,
                vmax=self.vmax_v,
                n_steps=self.n,
                alpha_mode=self.alpha_mode,
                tol=self.tol,
                file=self.file,
            )

    def _draw(self, s):
        if not self.live:
            return
        pad = max(0, self._last_len - len(s))
        self.file.write("\r" + s + " " * pad)
        self.file.flush()
        self._last_len = len(s)


class _CallableProgress:
    """Adapts a plain callable ``fn(i, info_dict)`` to the update() contract."""

    def __init__(self, fn):
        self._fn = fn

    def update(self, i, **info):
        self._fn(i, info)

    def close(self):
        pass


class DebugLog:
    """NDJSON trace of an IV sweep + every Newton iteration, for post-hoc
    debugging/investigation.  Drop-in ``progress`` for ``simulate``:

        from driftjax import console
        res = dj.sweep(des, ls, ..., progress=console.DebugLog("run.jsonl",
                           n_steps=41, vmax_v=1.1))

    One JSON object per line — load with ``pd.read_json(path, lines=True)``
    or filter with grep/jq.  Events:

      {"event": "sweep", "label", "n_steps", "vmax_v"}     (header)
      {"event": "step",  "i", "v_v", "iters", "resid", "backend",
       "fallback", "converged", "dt", "compiling"}          (per bias step)
      {"event": "iter",  "i", "phase", "it", "err", "resid"}
                                                             (per Newton iter)
      {"event": "done"}                                      (on close)

    Never affects numerics; missing fields are simply omitted.
    """

    def __init__(
        self, path, *, label="IV sweep", n_steps=None, vmax_v=None, append=False, run=None
    ):
        import json
        import os
        import time

        self.path = path
        self._json = json
        self._f = open(path, "a" if append else "w")
        self._i = None
        self._n = 0
        self._run = run
        self._closed = False
        if n_steps is not None:
            self._emit(
                {
                    "event": "sweep",
                    "label": label,
                    "run": run,
                    "pid": os.getpid(),
                    "t0": time.time(),
                    "n_steps": n_steps,
                    "vmax_v": vmax_v,
                }
            )

    # -- progress contract (called by solvers/continuation.sweep) ----------
    def update(self, i, **info):
        self._i = i
        self._emit({"event": "step", "i": i, "run": self._run, **info})

    def iter(self, it, err=None, resid=None, phase=None):
        self._emit(
            {
                "event": "iter",
                "i": self._i,
                "run": self._run,
                "phase": phase,
                "it": it,
                "err": err,
                "resid": resid,
            }
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._emit({"event": "done", "run": self._run})
        self._f.flush()
        self._f.close()

    # -- helpers ------------------------------------------------------------
    def _emit(self, rec):
        rec = {k: v for k, v in rec.items() if v is not None}
        self._f.write(self._json.dumps(rec, default=float) + "\n")
        self._n += 1

    @property
    def records(self):
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
