"""
ICC(A,1) is calculated per Schaefer parcel across participants.
A parcel value is only retained when it had finite values in all modeling strategies, otherwise, that parcel is excluded for that individual.
Parcel ICCs are then averaged.

Childhood = [5.5, 10.5)
Adolescence = [10.5, 17.5]
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RESULTS_DIR = Path("")
EXPORT_FOLDERS = {"individual": [],"pediatric": [],"adult": [],}
AGE_TABLES = [{"": Path(""},]
OUTPUT_DIR = Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ("individual", "pediatric", "adult")
BIOMARKERS = ("AbsolutePower", "RelativePower", "DFA", "fEI")
BIOMARKER_DISPLAY = {
    "AbsolutePower": "Absolute Power",
    "RelativePower": "Relative Power",
    "DFA": "DFA",
    "fEI": "fE/I",
}
BANDS = ("1-4 Hz", "4-8 Hz", "8-13 Hz", "13-30 Hz", "30-45 Hz")
GROUPS = ("Childhood", "Adolescence")
EXPECTED_PARCELS = 100
BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 20260913
SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)", re.I)
NDAR_RE = re.compile(r"NDAR[A-Za-z0-9]+", re.I)
FILE_RE = re.compile(
    r"^(?P<family>AbsolutePower|RelativePower|DFA|fEI)_"
    r"(?P<low>\d+(?:\.\d+)?)-(?P<high>\d+(?:\.\d+)?)Hz\.csv$",
    re.I,
)
METADATA_COLUMNS = {"age", "physical_age_years", "sex", "gender"}


def subject_id(value):
    text = str(value).strip()
    m = NDAR_RE.search(text)
    if m:
        return "sub-" + m.group().upper()
    m = SUBJECT_RE.search(text)
    if m:
        return m.group()
    raise ValueError(f"Unrecognized subject ID: {value!r}")


def merge_rows(frames):
    data = pd.concat(frames)
    rows = []
    for sid, group in data.groupby(level=0, sort=True):
        if (group.nunique(dropna=True) > 1).any():
            raise ValueError(f"Conflicting duplicate values for {sid}")
        rows.append(group.bfill().iloc[0].rename(sid))
    return pd.DataFrame(rows)


def read_matrix(path):
    data = pd.read_csv(path, index_col=0, low_memory=False)
    data.index = [subject_id(x) for x in data.index]
    data = data.drop(columns=[c for c in data.columns if c.lower() in METADATA_COLUMNS])
    if len(data.columns) != EXPECTED_PARCELS:
        raise ValueError(f"{path}: expected 100 parcels, found {len(data.columns)}")
    return data.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)


def load_data():
    collected = {}
    all_ids = set()
    parcels = None

    for strategy, folders in EXPORT_FOLDERS.items():
        by_condition = {}
        for folder in folders:
            for path in sorted(folder.rglob("*.csv")):
                m = FILE_RE.match(path.name)
                if m is None:
                    continue
                family = next(x for x in BIOMARKERS if x.lower() == m["family"].lower())
                band = f"{float(m['low']):g}-{float(m['high']):g} Hz"
                if band not in BANDS:
                    continue
                matrix = read_matrix(path)
                if parcels is None:
                    parcels = list(matrix.columns)
                if set(matrix.columns) != set(parcels):
                    raise ValueError(f"Parcel labels differ in {path}")
                all_ids.update(matrix.index)
                by_condition.setdefault((family, band), []).append(matrix[parcels])

        for family in BIOMARKERS:
            for band in BANDS:
                collected[strategy, family, band] = merge_rows(by_condition[family, band])

    ids = sorted(all_ids)

    age_frames = []
    for spec in AGE_TABLES:
        path = spec["path"]
        table = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        age_frames.append(
            pd.DataFrame(
                {"physical_age_years": pd.to_numeric(table[spec["age_column"]], errors="raise").to_numpy()},
                index=table[spec["id_column"]].map(subject_id),
            )
        )

    ages = merge_rows(age_frames).reindex(ids)
    age = ages["physical_age_years"]
    ages["group"] = np.select(
        [age.ge(5.5) & age.lt(10.5), age.ge(10.5) & age.le(17.5)],
        GROUPS,
        default="outside",
    )

    arrays = {}
    for family in BIOMARKERS:
        for band in BANDS:
            arrays[family, band] = np.stack(
                [collected[s, family, band].reindex(ids).to_numpy(float) for s in STRATEGIES],
                axis=-1,
            )
    return arrays, ages, parcels


# ICC(A,1)
def icc_a1(x, y):
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]

    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=0).astype(float)
    x = np.where(valid, x, 0.0)
    y = np.where(valid, y, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = x.sum(axis=0) / n
        mean_y = y.sum(axis=0) / n
        grand = (x.sum(axis=0) + y.sum(axis=0)) / (2 * n)
        row_mean = (x + y) / 2

        ssr = 2 * np.where(valid, (row_mean - grand) ** 2, 0).sum(axis=0)
        ssc = n * ((mean_x - grand) ** 2 + (mean_y - grand) ** 2)
        sst = np.where(valid, (x - grand) ** 2 + (y - grand) ** 2, 0).sum(axis=0)
        sse = sst - ssr - ssc

        msr = ssr / (n - 1)
        msc = ssc
        mse = sse / (n - 1)

        denominator = msr + mse + 2 * (msc - mse) / n
        icc = (msr - mse) / denominator

    icc[(n < 2) | ~np.isfinite(icc) | (denominator == 0)] = np.nan
    return icc


def parcel_iccs(x):
    # same subject × parcel cell is retained for all three strategies
    joint = np.isfinite(x).all(axis=2)
    x = np.where(joint[:, :, None], x, np.nan)

    pediatric = icc_a1(x[:, :, 0], x[:, :, 1])
    adult = icc_a1(x[:, :, 0], x[:, :, 2])

    # mean is calculated only over parcel ICCs finite for both comparisons
    matched = np.isfinite(pediatric) & np.isfinite(adult)
    return pediatric, adult, matched, joint, x


def mean_icc_pair(x):
    ped, adult, matched, _, _ = parcel_iccs(x)
    if not matched.any():
        return np.nan, np.nan, 0
    return float(np.mean(ped[matched])), float(np.mean(adult[matched])), int(matched.sum())


def bootstrap_difference_p(x, seed):
    # subjects with no jointly finite parcel do not enter the final analysis cohort
    eligible = np.isfinite(x).all(axis=2).any(axis=1)
    x = x[eligible]
    ped, adult, _ = mean_icc_pair(x)
    observed = ped - adult

    rng = np.random.default_rng(seed)
    draws = np.full(BOOTSTRAP_RESAMPLES, np.nan)
    for b in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, len(x), size=len(x))
        bp, ba, _ = mean_icc_pair(x[sampled])
        draws[b] = bp - ba

    valid = draws[np.isfinite(draws)]
    if len(valid) < int(np.ceil(0.95 * BOOTSTRAP_RESAMPLES)) or not np.isfinite(observed):
        return observed, np.nan

    # two-sided bootstrap p-value
    p = (1 + np.sum(np.abs(valid - observed) >= np.abs(observed))) / (len(valid) + 1)
    return observed, float(p)


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    work = np.where(finite, p, 1.0)
    order = np.argsort(work)
    m = len(work)
    adjusted = np.minimum.accumulate(
        (work[order] * m / np.arange(1, m + 1))[::-1]
    )[::-1]
    q = np.empty(m)
    q[order] = np.minimum(adjusted, 1.0)
    q[~finite] = np.nan
    return q


def calculate_summary(arrays, ages):
    rows = []
    seeds = np.random.SeedSequence(RANDOM_SEED)

    for family in BIOMARKERS:
        for band in BANDS:
            raw = arrays[family, band]
            for group in GROUPS:
                group_mask = ages["group"].eq(group).to_numpy()
                x = raw[group_mask]
                ped_mean, adult_mean, n_parcels = mean_icc_pair(x)
                delta, p = bootstrap_difference_p(x, seeds.spawn(1)[0])
                rows.append({
                    "group": group,
                    "biomarker": family,
                    "band": band,
                    "n_subjects": int(group_mask.sum()),
                    "finite_parcels": n_parcels,
                    "pediatric_icc": ped_mean,
                    "adult_icc": adult_mean,
                    "delta_icc": delta,
                    "p_bootstrap": p,
                })

    summary = pd.DataFrame(rows)
    # FDR (q < 0.05)
    # 2 age groups × 4 biomarkers × 5 bands = 40 tests
    summary["q_fdr"] = bh_fdr(summary["p_bootstrap"].to_numpy())
    summary["significant"] = summary["q_fdr"] < 0.05
    return summary


# results firgure 2A
def plot_icc_heatmap(summary):
    fig, axes = plt.subplots(2, 4, figsize=(20, 7), layout="constrained")

    for gi, group in enumerate(GROUPS):
        for fi, family in enumerate(BIOMARKERS):
            ax = axes[gi, fi]
            d = summary[(summary.group == group) & (summary.biomarker == family)].set_index("band").reindex(BANDS)
            matrix = np.vstack([d.pediatric_icc.to_numpy(), d.adult_icc.to_numpy()])
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

            for i in range(2):
                for j in range(5):
                    if np.isfinite(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                                fontweight="bold" if i == 0 else "normal")

            ax.set_xticks(range(5), BANDS, rotation=30, ha="right")
            ax.set_yticks([0, 1], ["Pediatric", "Adult"])
            ax.set_title(BIOMARKER_DISPLAY[family])
            if fi > 0:
                ax.set_yticklabels([])

        n = int(summary.loc[summary.group == group, "n_subjects"].iloc[0])
        axes[gi, 1].text(1.0, 1.28, f"{group} | N={n}", transform=axes[gi, 1].transAxes,
                         ha="center", fontsize=16)

    fig.suptitle(
        "Mean of parcel ICCs | ICC(A,1)\n"
        "Age-specific Template and Adult Template compared with participant-specific MRI model",
        fontweight="bold", fontsize=18,
    )
    fig.colorbar(image, ax=axes, shrink=0.65, label="ICC")
    fig.savefig(OUTPUT_DIR / "mean_parcel_icc_children_adolescence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# results firgure 2B
CATEGORY_SPECS = {
    "Poor": {"biomarker": "AbsolutePower", "low": -np.inf, "high": 0.50, "target": 0.26},
    "Moderate": {"biomarker": "DFA", "low": 0.50, "high": 0.75, "target": 0.61},
    "Good": {"biomarker": "fEI", "low": 0.75, "high": 0.90, "target": 0.83},
}


def build_example_candidates(arrays, ages, parcels):
    candidates, points = [], []

    for family in BIOMARKERS:
        for band in BANDS:
            raw = arrays[family, band]
            for group in GROUPS:
                group_mask = ages.group.eq(group).to_numpy()
                subject_ids = ages.index[group_mask]
                ped_icc, adult_icc, _, joint, x = parcel_iccs(raw[group_mask])

                # the parcel with the highest Pediatric-vs-Individual Pearson r
                ranking = []
                for p, parcel in enumerate(parcels):
                    mask = joint[:, p]
                    ind = x[mask, p, 0]
                    ped = x[mask, p, 1]
                    if len(ind) >= 3 and np.ptp(ind) > 0 and np.ptp(ped) > 0:
                        r = float(pearsonr(ind, ped).statistic)
                        ranking.append((p, parcel, len(ind), r))

                if not ranking:
                    continue

                p, parcel, _, selection_r = sorted(
                    ranking, key=lambda z: (-z[3], -z[2], str(z[1]))
                )[0]

                for comparison, j, values in (
                    ("pediatric_vs_individual", 1, ped_icc),
                    ("adult_vs_individual", 2, adult_icc),
                ):
                    mask = joint[:, p]
                    ind = x[mask, p, 0]
                    template = x[mask, p, j]
                    result = pearsonr(ind, template) if len(ind) >= 3 and np.ptp(ind) > 0 and np.ptp(template) > 0 else None
                    r = float(result.statistic) if result is not None else np.nan
                    pvalue = float(result.pvalue) if result is not None else np.nan

                    candidates.append({
                        "biomarker": family, "band": band, "group": group,
                        "comparison": comparison, "parcel": parcel, "parcel_index": p,
                        "n": len(ind), "parcel_icc": float(values[p]),
                        "pearson_r": r, "pearson_p": pvalue,
                        "selection_pediatric_r": selection_r,
                    })

                    for sid, xx, yy in zip(subject_ids[mask], ind, template):
                        points.append({
                            "subject": sid, "biomarker": family, "band": band,
                            "group": group, "comparison": comparison, "parcel": parcel,
                            "individual": xx, "template": yy,
                        })

    return pd.DataFrame(candidates), pd.DataFrame(points)


def choose_examples(candidates):
    chosen = []
    for label, spec in CATEGORY_SPECS.items():
        d = candidates[candidates.biomarker.eq(spec["biomarker"])].copy()
        d = d[np.isfinite(d.parcel_icc) & d.parcel_icc.ge(spec["low"]) & d.parcel_icc.lt(spec["high"])]
        if d.empty:
            raise RuntimeError(f"No eligible {label} example")
        d["distance_to_target"] = (d.parcel_icc - spec["target"]).abs()
        d = d.sort_values(
            ["distance_to_target", "selection_pediatric_r", "n", "biomarker", "band", "group", "parcel", "comparison"],
            ascending=[True, False, False, True, True, True, True, True],
        )
        row = d.iloc[0].copy()
        row["category"] = label
        chosen.append(row)
    return pd.DataFrame(chosen)


def plot_examples(selected, points):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, label in zip(axes, ("Poor", "Moderate", "Good")):
        rec = selected[selected.category == label].iloc[0]
        mask = (
            points.biomarker.eq(rec.biomarker)
            & points.band.eq(rec.band)
            & points.group.eq(rec.group)
            & points.comparison.eq(rec.comparison)
            & points.parcel.eq(rec.parcel)
        )
        d = points.loc[mask]
        x = d.individual.to_numpy()
        y = d.template.to_numpy()
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.08 * (high - low) if high > low else 1.0

        ax.scatter(x, y, alpha=0.65)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--")
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\n(ICC = {rec.parcel_icc:.2f} | r = {rec.pearson_r:.2f})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("ICC interpretation", fontweight="bold", fontsize=18)
    fig.supylabel("Template")
    fig.savefig(OUTPUT_DIR / "three_category_icc_examples_horizontal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(OUTPUT_DIR / "selected_icc_examples.csv", index=False)


def main():
    arrays, ages, parcels = load_data()
    summary = calculate_summary(arrays, ages)
    summary.to_csv(OUTPUT_DIR / "icc_summary.csv", index=False)
    plot_icc_heatmap(summary)

    candidates, points = build_example_candidates(arrays, ages, parcels)
    selected = choose_examples(candidates)
    plot_examples(selected, points)


if __name__ == "__main__":
    main()
