"""
For each participant and BEM surface:
1. read mesh vertices
2. calculate the geometric centroid
3. calculate every vertex-to-centroid Euclidean distance;=
4. calculate the mean centroid radius
5. compare Pediatric and Adult template errors with the participant-specific MRI

   Delta = |Pediatric - Individual| - |Adult - Individual|

Negative Delta means the pediatric template is closer.
The figure shows the mean Delta within each assigned pediatric-template age and a participant-level bootstrap 95% confidence interval.
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


INPUT_CSV = Path("")
PEDIATRIC_ROOT = Path("")
ADULT_ROOT = Path("")
OUTPUT_DIR = Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SURFACES = ("inner_skull", "outer_skull", "outer_skin")
SURFACE_LABELS = {
    "inner_skull": "Inner skull",
    "outer_skull": "Outer skull",
    "outer_skin": "Outer skin",
}
ASSIGNED_AGES = tuple(range(6, 18))
AGE_BOUNDARY = 10.5
EXPECTED_N = 384
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 20260922


def template_age(template_subject):
    match = re.search(r"ANTS(\d+)-0Years3T", str(template_subject))
    if match is None:
        raise ValueError(f"Cannot extract age from {template_subject!r}")
    return int(match.group(1))


def mean_centroid_radius(surface_path):
    """Mean Euclidean distance from mesh vertices to their geometric centroid"""
    vertices, _ = mne.read_surface(str(surface_path), read_metadata=False)
    vertices = np.asarray(vertices, dtype=float)

    centroid = vertices.mean(axis=0)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    return float(np.mean(distances))


def calculate_subject_deltas():
    manifest = pd.read_csv(INPUT_CSV)
    required = {
        "subject",
        "surface",
        "template_subject",
        "individual_surface_path",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    if manifest.subject.nunique() != EXPECTED_N:
        raise ValueError(
            f"Expected N={EXPECTED_N}; found {manifest.subject.nunique()} participants"
        )

    # each participant should have one row per BEM surface
    manifest = manifest[manifest.surface.isin(SURFACES)].copy()

    pediatric_cache = {}
    adult_cache = {}
    rows = []

    for _, row in manifest.iterrows():
        subject = row["subject"]
        surface = row["surface"]
        template = row["template_subject"]

        individual_path = Path(row["individual_surface_path"])
        pediatric_path = PEDIATRIC_ROOT / template / "bem" / f"{surface}.surf"
        adult_path = ADULT_ROOT / "bem" / f"{surface}.surf"

        individual_radius = mean_centroid_radius(individual_path)

        ped_key = (template, surface)
        if ped_key not in pediatric_cache:
            pediatric_cache[ped_key] = mean_centroid_radius(pediatric_path)
        pediatric_radius = pediatric_cache[ped_key]

        if surface not in adult_cache:
            adult_cache[surface] = mean_centroid_radius(adult_path)
        adult_radius = adult_cache[surface]

        pediatric_error = abs(pediatric_radius - individual_radius)
        adult_error = abs(adult_radius - individual_radius)

        rows.append(
            {
                "subject": subject,
                "surface": surface,
                "template_subject": template,
                "assigned_template_age": template_age(template),
                "individual_mean_radius_mm": individual_radius,
                "pediatric_mean_radius_mm": pediatric_radius,
                "adult_mean_radius_mm": adult_radius,
                "pediatric_abs_error_mm": pediatric_error,
                "adult_abs_error_mm": adult_error,
                "delta_mm": pediatric_error - adult_error,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "bem_mean_centroid_radius_subject_values.csv", index=False)
    return result


def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    observed = float(np.mean(values))
    if len(values) == 1:
        return observed, observed, observed

    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES)

    for b in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[b] = np.mean(sample)

    return (
        observed,
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def summarize_by_age(data):
    rows = []
    seed_sequence = np.random.SeedSequence(RANDOM_SEED)

    for surface in SURFACES:
        for age in ASSIGNED_AGES:
            values = data.loc[
                (data.surface == surface)
                & (data.assigned_template_age == age),
                "delta_mm",
            ].to_numpy(float)

            mean, low, high = bootstrap_mean_ci(
                values,
                seed_sequence.spawn(1)[0],
            )

            rows.append(
                {
                    "surface": surface,
                    "assigned_template_age": age,
                    "n": int(np.isfinite(values).sum()),
                    "mean_delta_mm": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "bem_mean_centroid_radius_by_age.csv", index=False)
    return summary


def plot_delta(summary):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    for surface in SURFACES:
        d = (
            summary[summary.surface == surface]
            .set_index("assigned_template_age")
            .reindex(ASSIGNED_AGES)
        )
        x = np.asarray(ASSIGNED_AGES)
        y = d.mean_delta_mm.to_numpy()
        low = d.ci_low.to_numpy()
        high = d.ci_high.to_numpy()

        line = ax.plot(x, y, marker="o", linewidth=2, label=SURFACE_LABELS[surface])[0]
        ax.fill_between(x, low, high, alpha=0.15, color=line.get_color())

    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.axvline(AGE_BOUNDARY, linestyle=":", linewidth=2)

    ax.text(8, 1.01, "Childhood", transform=ax.get_xaxis_transform(), ha="center", va="bottom")
    ax.text(14, 1.01, "Adolescence", transform=ax.get_xaxis_transform(), ha="center", va="bottom")

    ax.set_xticks(ASSIGNED_AGES)
    ax.set_xlabel("Assigned pediatric template age (years)")
    ax.set_ylabel("Δ (mm): |PED − IND| − |ADULT − IND|")
    ax.set_title(f"Mean centroid radius (mm) | N = {EXPECTED_N}", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "mean_centroid_radius_delta.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    subject_values = calculate_subject_deltas()
    summary = summarize_by_age(subject_values)
    plot_delta(summary)


if __name__ == "__main__":
    main()
