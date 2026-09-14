"""Developer contract for JIT warm-up, batched solves, and optics sharding."""

import time

import numpy as np

import driftjax as dj
from driftjax.science.optics import generation_profile
from driftjax.solvers.batched import sweep_batched
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    example_args,
    execution_metadata,
    new_figure,
    pn_device,
    report,
    report_fom,
    run_timed,
    save_figure,
    save_json,
    si_material,
)


def main():
    example_args("developer_batching_and_jit")
    points = 500
    steps = 61
    device = pn_device(si_material(), points=points)
    serial, first_s = run_timed(dj.simulate, device, dj.Sweep(vmax=0.75, n_steps=steps))
    repeat, warm_s = run_timed(dj.simulate, device, dj.Sweep(vmax=0.75, n_steps=steps))
    design = device.design()
    cell = dj.simulator.init_cell(design, dj.AM15G(), alpha_mode=device.alpha_mode)
    batch_start = time.perf_counter()
    vb, jb, _ = sweep_batched(cell, vmax=0.75 / float(dj.energy), n_steps=steps)
    jb.block_until_ready()
    batch_s = time.perf_counter() - batch_start  # includes one-time compile
    v_batch = np.asarray(vb) * float(dj.energy)
    j_batch = np.asarray(jb) * float(dj.current)
    rel_current = float(
        np.max(np.abs(j_batch - np.asarray(serial.currents))) / np.max(np.abs(serial.currents))
    )
    full_generation = generation_profile(design, dj.AM15G(), alpha_mode=device.alpha_mode)
    sharded, shard_meta = dj.sharded_generation(design, dj.AM15G(), n_shards=2)
    rel_generation = float(
        np.max(np.abs(sharded - full_generation)) / np.max(np.abs(full_generation))
    )

    fig, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    axes[0].plot(
        serial.voltages, serial.currents * 1e3, color=style.BLUE, linewidth=1.6, label="serial API"
    )
    axes[0].plot(v_batch, j_batch * 1e3, "o", ms=5, color=style.ORANGE, label="batched solver")
    axes[0].set(xlabel=style.LBL_BIAS, ylabel="current density / mA cm$^{-2}$")
    axes[0].set_xlim(0, 0.75)
    jsc_vals = [
        abs(float(serial.jsc) * 1e3),
        abs(float(j_batch[0]) * 1e3) if len(j_batch) > 0 else 15,
    ]
    y_top = max(jsc_vals)
    axes[0].set_ylim(0, y_top * 1.15)
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    axes[0].legend(frameon=False, fontsize=8, handlelength=1.2)
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[0].set_title(f"Batched vs serial (rel err {rel_current:.1e})", fontsize=9, pad=8)
    axes[1].bar(
        ["first", "warm", "batch"],
        [first_s, warm_s, batch_s],
        color=(style.BLUE, style.GREEN, style.VIOLET),
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    axes[1].set(xlabel="execution path", ylabel="wall time / s")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", which="both", alpha=0.18, linestyle="--")
    axes[1].set_title("JIT warm-up vs batched (log)", fontsize=9, pad=8)
    figure = save_figure(fig, "developer_batching_and_jit")
    save_json(
        "developer_batching_and_jit",
        {
            "metadata": execution_metadata(),
            "first_call_s": first_s,
            "warm_call_s": warm_s,
            "batched_call_s": batch_s,
            "max_relative_current_error": rel_current,
            "max_relative_generation_error": rel_generation,
            "sharding": shard_meta,
            "repeat_efficiency_delta": abs(float(serial.efficiency - repeat.efficiency)),
            "figure": figure.name,
        },
    )
    report(
        "developer_batching_and_jit",
        iv_rel_error=float(rel_current),
        generation_rel_error=float(rel_generation),
    )

    report_fom("developer_batching_and_jit", serial=serial, repeat=repeat)


if __name__ == "__main__":
    main()
