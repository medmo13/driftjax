"""Matplotlib renderers for inverse-design / optimizer convergence diagnostics.

These render convergence curves and design-parameter trajectories in the
shared ``driftjax.viz`` presentation style. Use them after an optimization
loop that has recorded the objective (and, optionally, the design vector) each
step.
"""
import matplotlib

# Do not force Agg globally if caller already selected a backend (e.g. notebook).
# Only set Agg when no backend has been chosen (matplotlib default is Agg anyway).
if matplotlib.get_backend().lower() == "agg" or "matplotlib_inline" not in str(matplotlib.get_backend()).lower():
    try:
        # Use Agg only if no explicit non-Agg backend is active; safe to call early.
        if matplotlib.rcParams.get("backend", "Agg") == "Agg":
            matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import numpy as np

from driftjax.viz import style


def convergence(values, path, ylabel="objective", title=None,
               xlabel="optimizer iteration", final_line=True, yscale=None):
    """Plot the scalar objective across optimizer evaluations.

    ``values`` may be a 1-D sequence (single curve) or a dict mapping labels
    to sequences (multiple curves, e.g. one per optimizer backend).
    """
    style.apply("presentation")
    fig, ax = plt.subplots(figsize=(8, 5))
    if isinstance(values, dict):
        for name, seq in values.items():
            seq = np.asarray(seq, float)
            ax.plot(np.arange(1, len(seq) + 1), seq, "-o", ms=3, lw=1.5, label=name)
        ax.legend(fontsize=10)
    else:
        seq = np.asarray(values, float)
        ax.plot(np.arange(1, len(seq) + 1), seq, "k-o", ms=3, lw=1.5)
        if final_line and len(seq):
            ax.axhline(seq[-1], color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def parameters(traj, path, labels=None, targets=None, title=None,
               ylabel="design variable", xlabel="optimizer iteration"):
    """Plot each design coordinate vs optimizer iteration (with target markers)."""
    traj = np.asarray(traj, float)
    if traj.ndim == 1:
        traj = traj[:, None]
    style.apply("presentation")
    fig, ax = plt.subplots(figsize=(8, 5))
    n = traj.shape[1]
    for i in range(n):
        ax.plot(np.arange(1, len(traj) + 1), traj[:, i], lw=1.5,
                label=(labels[i] if labels else f"x{i}"))
        if targets is not None and i < len(targets):
            ax.axhline(float(targets[i]), color="k", ls=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend(fontsize=9, ncol=2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
