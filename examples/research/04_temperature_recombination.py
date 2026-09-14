"""Research example 4: temperature dependence under explicit assumptions.

Question: how does the simulated device respond to temperature when all
material parameters except the thermal scaling are held fixed? The panel
reports Voc(T) and efficiency(T); the fitted dVoc/dT is recorded in the JSON
sidecar. (The textbook -2 mV/K additionally assumes Eg(T) and Nc,v ∝ T^3/2,
which this first-order model intentionally omits.)
"""

import numpy as np

import driftjax as dj
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    example_args,
    execution_metadata,
    new_figure,
    pn_device,
    report,
    save_figure,
    save_json,
    si_material,
    solution_metrics,
)


def main():
    example_args("04_temperature_recombination")
    device = pn_device(si_material(), points= 500)
    temperatures = (280.0, 300.0, 320.0, 340.0)
    solutions = {
        temperature: dj.simulate(
            device, dj.Sweep(vmax=0.85, n_steps= 25), T=temperature
        )
        for temperature in temperatures
    }
    metrics = {str(T): solution_metrics(solution) for T, solution in solutions.items()}
    voc_series = [solutions[T].voc for T in temperatures]
    voc_slope_mV_per_K = float(np.polyfit(temperatures, voc_series, 1)[0] * 1e3)

    figure, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    axes[0].plot(temperatures, voc_series, "o-", color=style.BLUE, markersize=5, linewidth=1.4)
    axes[0].set(xlabel="temperature / K", ylabel="$V_{oc}$ / V")
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[0].set_title(f"$dV_{{oc}}/dT$ = {voc_slope_mV_per_K:.2f} mV/K", fontsize=9, pad=8)
    # Annotate slope
    axes[0].text(0.05, 0.95, r"fixed $E_g$, $N_{c,v}$, $\mu$" "\n(Vt scaling only)", transform=axes[0].transAxes, va="top", fontsize=6.5, style="italic", color=style.GRAY,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    axes[1].plot(
        temperatures,
        [solutions[T].efficiency * 100 for T in temperatures],
        "o-",
        color=style.RED, markersize=5, linewidth=1.4,
    )
    axes[1].set(xlabel="temperature / K", ylabel="efficiency / %")
    axes[1].grid(alpha=0.18, linestyle="--")
    axes[1].set_title("Efficiency vs $T$ (first-order model)", fontsize=9, pad=8)
    figure_path = save_figure(figure, "research_04_temperature_recombination")
    save_json("research_04_temperature_recombination", {
        "metadata": execution_metadata(),
        "assumption": "material parameters are held fixed except thermal scaling",
        "metrics": metrics,
        "voc_slope_mV_per_K": voc_slope_mV_per_K,
        "figure": figure_path.name,
    })
    report("research_04", voc_slope_mV_per_K=voc_slope_mV_per_K)


if __name__ == "__main__":
    main()
