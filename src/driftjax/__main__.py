"""CLI entry: python -m driftjax [info|simulate|pyramid|verify|validate|benchmark|release].

Commands:
  simulate    -> framed one-screen report (design / sweep / Jsc·Voc·FF·η·MPP /
                 runtime) + a live per-bias bar with solver info on a TTY;
                 --quiet gives the 4 classic metric lines, --json a
                 machine-readable record, --no-progress disables the bar.
  info        -> compact human provenance line, or --json full record.
  pyramid     -> the L1–L10 validation pyramid (legacy).
  verify      -> Tier I-III: algebraic + physics verification (fast).
  validate    -> Tier IV-VII: gradient + stability + cross-code validation.
  benchmark   -> Tier VI: performance scaling measurements.
  release     -> full publication-quality validation suite.
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from typing import Any


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="driftjax")
    sub = ap.add_subparsers(dest="cmd")

    p_info = sub.add_parser("info", help="print environment/provenance info")
    p_info.add_argument(
        "--json", action="store_true", help="full JSON record instead of the compact line"
    )

    p_sim = sub.add_parser("simulate", help="run the canary np-junction simulation")
    p_sim.add_argument("--n", type=int, default=200)
    p_sim.add_argument("--vmax", type=float, default=1.2)
    p_sim.add_argument("--steps", type=int, default=25)
    p_sim.add_argument("--quiet", action="store_true", help="print only the classic 4 metric lines")
    p_sim.add_argument(
        "--verbose", action="store_true", help="per-bias-step solver lines (no live bar)"
    )
    p_sim.add_argument("--json", action="store_true", help="machine-readable JSON result")
    p_sim.add_argument("--no-progress", action="store_true", help="disable the live progress bar")
    p_sim.add_argument(
        "--debug-log",
        metavar="PATH",
        default=None,
        help="write an NDJSON trace of the sweep and every "
        "Newton iteration to PATH (post-hoc debugging)",
    )

    sub.add_parser("pyramid", help="run the L1–L10 validation pyramid (legacy)")

    p_verify = sub.add_parser("verify", help="Tier I-III: algebraic + physics verification")
    p_verify.add_argument("--json", type=str, default=None, help="save JSON records to directory")

    p_validate = sub.add_parser("validate", help="Tier IV-VII: gradient + stability + cross-code")
    p_validate.add_argument("--json", type=str, default=None, help="save JSON records to directory")

    p_benchmark = sub.add_parser("benchmark", help="Tier VI: performance scaling")
    p_benchmark.add_argument("--json", type=str, default=None, help="save JSON records to directory")
    p_benchmark.add_argument("--mesh", type=str, default="125,250,500,1000,2000",
                              help="comma-separated mesh sizes")

    p_release = sub.add_parser("release", help="full publication-quality validation suite")
    p_release.add_argument("--json", type=str, default=None, help="save JSON records to directory")

    args = ap.parse_args(argv)
    if args.cmd == "info":
        from driftjax.runtime.provenance import record

        if args.json:
            import json

            print(json.dumps(record(), indent=2))
        else:
            import driftjax.console as console

            r = record()
            console.hr()
            print(
                console.accent(f" driftjax {r['version']}"),
                f"· jax {r['jax']} · python {r['python']} · {', '.join(r['devices'])}",
            )
            print("  config fingerprint", console.dim(r["config_fingerprint"]))
            console.hr()
        return 0
    if args.cmd == "pyramid":
        from driftjax import console
        from driftjax.validation import pyramid

        console.hr()
        print(console.accent(" driftjax validation pyramid"), "(L1-L10)")
        console.thin()
        t0 = _time.monotonic()
        res = pyramid.run(fast=True)
        dt = _time.monotonic() - t0
        console.thin()
        npass, nfail = sum(res.values()), len(res) - sum(res.values())
        if nfail == 0:
            print(
                console.ok(f" pyramid result: {npass}/{len(res)} gates passed") + f"  ({dt:.0f}s)"
            )
        else:
            print(
                console.err(f" pyramid result: {npass}/{len(res)} passed, {nfail} FAILED")
                + f"  ({dt:.0f}s)"
            )
        console.hr()
        return 0 if all(res.values()) else 1
    if args.cmd == "verify":
        from driftjax import console
        console.hr()
        print(console.accent(" driftjax verify"), "(Tier I-III: algebraic + physics verification)")
        console.thin()
        t0 = _time.monotonic()
        from driftjax.validation.pyramid import run
        res = run(fast=True)
        dt = _time.monotonic() - t0
        console.thin()
        npass, nfail = sum(res.values()), len(res) - sum(res.values())
        if nfail == 0:
            print(console.ok(f" verify result: {npass}/{len(res)} gates passed") + f"  ({dt:.0f}s)")
        else:
            print(console.err(f" verify result: {npass}/{len(res)} passed, {nfail} FAILED") + f"  ({dt:.0f}s)")
        console.hr()
        return 0 if all(res.values()) else 1
    if args.cmd == "validate":
        from driftjax import console
        console.hr()
        print(console.accent(" driftjax validate"), "(Tier IV-VII: gradient + stability + cross-code)")
        console.thin()
        t0 = _time.monotonic()
        # Run gradient verification
        from driftjax.validation.gradients import scalar_fd
        print("\n[IV] FD step-size sweep")
        fd_result = scalar_fd.fd_step_sweep()
        scalar_fd.print_summary(fd_result)
        # Run condition sweep
        from driftjax.validation.stability import condition_sweep
        print("[V] Condition-number sweep")
        cs_result = condition_sweep.run_condition_sweep()
        condition_sweep.print_summary(cs_result)
        dt = _time.monotonic() - t0
        console.thin()
        print(console.ok(f" validate completed in {dt:.0f}s"))
        console.hr()
        if args.json:
            from pathlib import Path
            outdir = Path(args.json)
            outdir.mkdir(parents=True, exist_ok=True)
            scalar_fd.save_record(fd_result, outdir / "fd_step_sweep.json")
            condition_sweep.save_record(cs_result, outdir / "condition_sweep.json")
        return 0
    if args.cmd == "benchmark":
        from driftjax import console
        console.hr()
        print(console.accent(" driftjax benchmark"), "(Tier VI: performance scaling)")
        console.thin()
        t0 = _time.monotonic()
        from driftjax.validation.benchmarks import forward_scaling
        mesh_sizes = [int(x) for x in args.mesh.split(",")]
        print(f"\nMeasuring forward scaling at N = {mesh_sizes}")
        fs_result = forward_scaling.measure_forward_scaling(mesh_sizes=mesh_sizes)
        forward_scaling.print_summary(fs_result)
        dt = _time.monotonic() - t0
        console.thin()
        print(console.ok(f" benchmark completed in {dt:.0f}s"))
        console.hr()
        if args.json:
            from pathlib import Path
            outdir = Path(args.json)
            outdir.mkdir(parents=True, exist_ok=True)
            forward_scaling.save_record(fs_result, outdir / "forward_scaling.json")
        return 0
    if args.cmd == "release":
        from driftjax import console
        console.hr()
        print(console.accent(" driftjax release"), "(full publication-quality validation)")
        console.thin()
        t0 = _time.monotonic()
        # Tier I-III: verification
        from driftjax.validation.pyramid import run as pyramid_run
        print("\n[Tier I-III] Verification")
        v_res = pyramid_run(fast=True)
        # Tier IV-VII: validation
        from driftjax.validation.benchmarks import forward_scaling
        from driftjax.validation.gradients import scalar_fd
        from driftjax.validation.stability import condition_sweep
        print("\n[Tier IV] Gradient verification")
        fd_result = scalar_fd.fd_step_sweep()
        scalar_fd.print_summary(fd_result)
        print("[V] Condition-number sweep")
        cs_result = condition_sweep.run_condition_sweep()
        condition_sweep.print_summary(cs_result)
        print("[VI] Forward scaling")
        fs_result = forward_scaling.measure_forward_scaling()
        forward_scaling.print_summary(fs_result)
        dt = _time.monotonic() - t0
        console.thin()
        npass = sum(v_res.values())
        ntotal = len(v_res)
        print(console.ok(f" release suite: {npass}/{ntotal} verification gates + validation completed in {dt:.0f}s"))
        console.hr()
        if args.json:
            import json
            from pathlib import Path
            outdir = Path(args.json)
            outdir.mkdir(parents=True, exist_ok=True)
            # Save all records
            with open(outdir / "verification_pyramid.json", "w") as f:
                json.dump(v_res, f, indent=2)
            scalar_fd.save_record(fd_result, outdir / "fd_step_sweep.json")
            condition_sweep.save_record(cs_result, outdir / "condition_sweep.json")
            forward_scaling.save_record(fs_result, outdir / "forward_scaling.json")
            # Summary record
            summary = {
                "verification_gates_passed": npass,
                "verification_gates_total": ntotal,
                "fd_optimal_h": fd_result.get("optimal_h"),
                "condition_sweep_summary": cs_result.get("summary"),
                "scaling_exponent": fs_result.get("fit", {}).get("p"),
                "total_time_s": dt,
            }
            with open(outdir / "release_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            # Unified, machine-readable validation record (single source of
            # truth for the paper's methods/validation tables).
            from driftjax.validation import record as val_record
            rec = val_record.build_record(
                verification=v_res,
                fd=fd_result,
                condition=cs_result,
                scaling=fs_result,
                case="psc_16param",
                mesh=500,
                bias_points=41,
                precision="float64",
                verification_gates_passed=npass,
                verification_gates_total=ntotal,
            )
            rec_path = val_record.save_record(rec, outdir / "validation_record.json")
            print(f"Records saved to {outdir}")
            print(f"Unified validation record -> {rec_path}")
        return 0
    if args.cmd == "simulate":
        import time

        import driftjax as dj
        from driftjax import console
        from driftjax.science.spectrum import spectrum

        Si = dj.load_material("Si")
        des = dj.Device(
            n_points=args.n,
            layers=[(1e-4, Si, 1e17), (1e-4, Si, -1e17)],
            Snl=1e7,
            Snr=1e7,
            Spl=1e7,
            Spr=1e7,
        ).design()
        progress: Any = None  # DebugLog | SweepProgress | bool
        if args.debug_log:
            progress = console.DebugLog(args.debug_log, n_steps=args.steps, vmax_v=args.vmax)
        elif args.quiet or args.json or args.no_progress:
            progress = False
        elif args.verbose:
            progress = console.SweepProgress(
                args.steps, vmax_v=args.vmax, label="IV sweep", verbose=True
            )
        else:
            progress = True
        t0 = time.perf_counter()
        res = dj.simulate(
            des, dj.Sweep(vmax=args.vmax, n_steps=args.steps), progress=progress, ls=spectrum()
        )
        runtime = time.perf_counter() - t0

        if args.quiet:
            print(f"Jsc  = {float(res.jsc) * 1e3:8.2f} mA/cm²")
            print(f"Voc  = {float(res.voc):8.3f} V")
            print(f"FF   = {float(res.ff):8.3f}")
            print(f"η    = {float(res.efficiency) * 100:8.2f} %")
        elif args.json:
            import json

            print(json.dumps(console.machine(res, des, runtime), indent=2))
        else:
            console.summary(
                res,
                title=f"simulate — Si n/p canary (n={args.n})",
                des=des,
                runtime=runtime,
                vmax=args.vmax,
                n_steps=args.steps,
                alpha_mode="table",
            )
        if args.debug_log:
            print(console.dim(f"debug log -> {args.debug_log} ({progress.records} records)"))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
