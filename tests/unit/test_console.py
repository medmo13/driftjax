"""Console-output unit tests.

* reports render correctly in non-TTY (piped) mode and stay ANSI-free;
* SweepProgress lifecycle is safe on non-TTY stdout (live bar is a TTY-only
  construct; verbose per-step table renders to any stream);
* progress=True never changes numerics (same result dict as silent run);
* CLI --json/--quiet behave as documented.
"""

import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import driftjax as dj
import driftjax.console as console
import driftjax.science
from driftjax.science.spectrum import spectrum

SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"
)


def _design(n=60):
    mat = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=100,
        Et=0,
        tn=1e-08,
        tp=1e-08,
        A=20000.0,
    )
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=n,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()


def _sim(n=60, progress=False):
    return dj.simulate(
        _design(n),
        dj.Sweep(vmax=0.8, n_steps=9),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        progress=progress,
        ls=spectrum(),
    )


@pytest.fixture(scope="module")
def sim9():
    """One shared 9-step solution for all rendering/API tests (was a fresh
    full sweep per test — 4+ solves for identical Solution content)."""
    return _sim()


class TestSummary:
    def test_renders_metrics_and_no_ansi(self, capsys, sim9):
        res = sim9
        console.summary(
            res,
            title="test run",
            des=_design(),
            runtime=1.23,
            vmax=0.8,
            n_steps=9,
            alpha_mode="beer-lambert",
        )
        out = capsys.readouterr().out
        assert "Jsc" in out and "Voc" in out and ("FF" in out)
        assert "η" in out and "MPP" in out and ("runtime" in out)
        assert "test run" in out and "n_points" in out
        assert "\x1b[" not in out

    def test_summary_accepts_dict_without_design(self, capsys):
        console.summary(
            {
                "jsc_density": 0.02,
                "voc_v": 1.0,
                "ff": 0.8,
                "eff": 0.18,
                "iv": ([0.0, 0.5], [-0.02, 0.0]),
            },
            title="bare",
        )
        out = capsys.readouterr().out
        assert "Jsc" in out and "MPP" in out

    def test_machine_json_roundtrip(self, sim9):
        res = sim9
        m = console.machine(res, _design(), runtime=3.5)
        blob = json.loads(json.dumps(m))
        voc = float(res.voc)
        if voc == voc:
            assert blob["voc_V"] == pytest.approx(voc)
        else:
            assert blob["voc_V"] is None
        if float(res.efficiency) == float(res.efficiency):
            assert blob["eff_pct"] == pytest.approx(float(res.efficiency) * 100)
        else:
            assert blob["eff_pct"] is None
        assert blob["runtime_s"] == 3.5
        assert len(blob["iv"]["v_V"]) == 9

    def test_truncated_sweep_shows_bound_and_na(self, capsys, sim9):
        """vmax below Voc: summary prints '> vmax' and FF n/a instead of garbage."""
        res = sim9
        assert float(res.voc) != float(res.voc)
        console.summary(
            res,
            title="truncated",
            des=_design(),
            runtime=1.0,
            vmax=0.8,
            n_steps=9,
            alpha_mode="beer-lambert",
        )
        out = capsys.readouterr().out
        assert "> 0.800 V" in out and "sweep below Voc" in out
        assert "n/a" in out and "η" in out


class TestSweepProgress:
    def test_quiet_non_tty_writes_nothing(self):
        buf = io.StringIO()
        p = console.SweepProgress(5, label="IV sweep", verbose=False, file=buf)
        p.update(0, v_v=0.1, iters=4, resid=1e-14, dt=0.2)
        p.update(1, v_v=0.2, iters=3, resid=2e-14, dt=0.1)
        p.close()
        assert buf.getvalue() == ""

    def test_verbose_table_renders(self):
        buf = io.StringIO()
        p = console.SweepProgress(3, verbose=True, file=buf)
        p.update(0, v_v=0.0, iters=6, resid=3e-14, dt=1.2)
        p.update(1, v_v=0.4, iters=4, resid=2.5e-14, dt=0.3)
        p.update(2, v_v=0.8, iters=2, resid=1e-14, dt=0.2, fallback="ls", converged=False)
        p.close()
        out = buf.getvalue()
        assert "bias/V" in out and "iters" in out
        assert "3.00e-14" in out and "2.50e-14" in out
        assert "converged" in out and "ls" in out

    def test_double_close_is_safe(self):
        buf = io.StringIO()
        p = console.SweepProgress(2, verbose=True, file=buf)
        p.update(0, v_v=0.0, iters=2, resid=1e-14, dt=0.1)
        p.close()
        p.close()


class TestNumericsUntouched:
    def test_progress_runs_identical_to_silent(self):
        r_silent = _sim(progress=False)
        r_live = _sim(progress=True)
        for attribute in ("jsc", "voc", "ff", "efficiency"):
            a, b = (float(getattr(r_silent, attribute)), float(getattr(r_live, attribute)))
            # Progress reporting must not perturb numerics beyond rounding:
            # quiet runs take the compiled-scan fast path while reporting runs
            # take the serial sweep; the two valid paths differ at ~1e-14
            # (op-ordering only). NaN==NaN still required on both sides.
            assert a == b or (a != a and b != b) or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)), (
                f"{attribute}: {a} != {b}"
            )
        np.testing.assert_array_equal(np.asarray(r_silent.voltages), np.asarray(r_live.voltages))
        # same rounding-level allowance as above (two valid execution paths)
        np.testing.assert_allclose(
            np.asarray(r_silent.current), np.asarray(r_live.current), rtol=1e-9, atol=0
        )

    def test_callable_progress_receives_steps(self):
        seen = []
        _sim(progress=lambda i, info: seen.append((i, info["v_v"], info["iters"])))
        assert len(seen) == 9
        assert seen[0][0] == 0 and seen[-1][0] == 8
        assert all((v is not None for _, v, _ in seen))


class TestCLI:
    ENV = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONPATH": SRC,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    CW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "driftjax", *args],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=self.CW,
            env=self.ENV,
        )

    @pytest.mark.slow  # fresh interpreter + JAX import + solve per subprocess
    def test_cli_json_parses(self):
        r = self.run_cli("simulate", "--n", "40", "--steps", "7", "--json")
        assert r.returncode == 0, r.stderr
        m = json.loads(r.stdout)
        assert {"jsc_density_A_per_cm2", "voc_V", "ff", "eff_pct", "runtime_s"} <= set(m)
        assert len(m["iv"]["v_V"]) == 7

    @pytest.mark.slow  # fresh interpreter + JAX import + solve per subprocess
    def test_cli_quiet_classic_lines(self):
        r = self.run_cli("simulate", "--n", "40", "--steps", "7", "--quiet")
        assert r.returncode == 0
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        assert lines[0].startswith("Jsc")
        assert lines[1].startswith("Voc")
        assert lines[2].startswith("FF")
        assert lines[3].startswith("η")
        assert "═" not in r.stdout and "Jsc" in lines[0]

    def test_cli_info_compact(self):
        r = self.run_cli("info")
        assert r.returncode == 0
        assert "driftjax" in r.stdout and "jax" in r.stdout


class TestDebugLog:
    def test_trace_has_sweep_step_iter_done(self, tmp_path):
        log = tmp_path / "run.jsonl"
        _sim(progress=console.DebugLog(str(log), n_steps=9, vmax_v=0.8))
        with open(log) as _f:
            lines = [json.loads(rec) for rec in _f if rec.strip()]
        events = [r["event"] for r in lines]
        assert events[0] == "sweep" and events[-1] == "done"
        assert lines[0]["pid"] == os.getpid() and "t0" in lines[0]
        assert events.count("step") == 9
        assert events.count("iter") >= 9
        step0 = next(r for r in lines if r["event"] == "step")
        assert {"i", "v_v", "iters", "resid", "converged", "dt"} <= set(step0)
        first_iter = next(r for r in lines if r["event"] == "iter")
        assert first_iter["phase"] == "equilibrium"
        iters = [r for r in lines if r["event"] == "iter"]
        assert any(r["phase"] and r["phase"].startswith("V = ") for r in iters)
        assert all(
            r["resid"] is not None and r["resid"] > 0
            for r in lines
            if r["event"] == "iter" and r["resid"] is not None
        )

    def test_debug_log_leaves_numerics_untouched(self, tmp_path):
        log = tmp_path / "run.jsonl"
        r_silent = _sim(progress=False)
        r_log = _sim(progress=console.DebugLog(str(log)))
        for attribute in ("jsc", "voc", "ff", "efficiency"):
            a, b = (float(getattr(r_silent, attribute)), float(getattr(r_log, attribute)))
            # Progress reporting must not perturb numerics beyond rounding:
            # quiet runs take the compiled-scan fast path while reporting runs
            # take the serial sweep; the two valid paths differ at ~1e-14
            # (op-ordering only). NaN==NaN still required on both sides.
            assert a == b or (a != a and b != b) or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)), (
                f"{attribute}: {a} != {b}"
            )
        np.testing.assert_allclose(
            np.asarray(r_silent.current), np.asarray(r_log.current), rtol=1e-9, atol=0
        )

    @pytest.mark.slow  # fresh interpreter + JAX import + solve per subprocess
    def test_cli_debug_log_writes_file(self, tmp_path):
        log = tmp_path / "cli.jsonl"
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "driftjax",
                "simulate",
                "--n",
                "40",
                "--steps",
                "7",
                "--debug-log",
                str(log),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=TestCLI.CW,
            env=TestCLI.ENV,
        )
        assert r.returncode == 0, r.stderr
        assert log.exists()
        with open(log) as _f:
            lines = [json.loads(rec) for rec in _f if rec.strip()]
        assert lines[0]["event"] == "sweep" and lines[-1]["event"] == "done"
        assert sum(1 for rec in lines if rec["event"] == "step") == 7
        assert "debug log ->" in r.stdout

    def test_append_runs_are_tagged(self, tmp_path):
        """One append-mode file, several sweeps -> every record carries run N."""
        log = tmp_path / "multi.jsonl"
        for run in (1, 2):
            _sim(progress=console.DebugLog(str(log), n_steps=9, vmax_v=0.8, append=True, run=run))
        with open(log) as _f:
            lines = [json.loads(rec) for rec in _f if rec.strip()]
        runs = {r["run"] for r in lines}
        assert runs == {1, 2}
        steps1 = [r for r in lines if r["event"] == "step" and r["run"] == 1]
        steps2 = [r for r in lines if r["event"] == "step" and r["run"] == 2]
        assert len(steps1) == 9 and len(steps2) == 9
        assert lines[0]["run"] == 1
        assert [r for r in lines if r["event"] == "done"] == [
            {"event": "done", "run": 1},
            {"event": "done", "run": 2},
        ]
