from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("")
EXPORT_FOLDERS = {"individual": [],"pediatric": [],"adult": [],}
AGE_TABLES = [{"": Path("")},
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
YEARS = tuple(range(6, 18))
NETWORKS = (
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Salience/Ventral Attention",
    "Limbic",
    "Control",
    "Default",
)
NETWORK_TOKENS = dict(
    zip(
        ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"),
        NETWORKS,
    )
)
EXPECTED_PARCELS = 100
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


def parcel_network(parcel):
    """map Schaefer parcels to the seven networks"""
    label = str(parcel)
    for token, network in NETWORK_TOKENS.items():
        if re.search(rf"(?:^|[_/]){re.escape(token)}(?:[_/-]|$)", label, re.I):
            return network

    short = dict(zip(("V", "SM", "DA", "SVA", "L", "C", "D"), NETWORKS))
    base = label.rsplit("-", 1)[0]
    for token in sorted(short, key=len, reverse=True):
        if re.match(rf"^{re.escape(token)}(?:[_/]|\d|$)", base, re.I):
            return short[token]

    raise ValueError(f"Cannot assign network for parcel {parcel!r}")


def load_data():
    collected = {}
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
                by_condition.setdefault((family, band), []).append(matrix[parcels])

        for family in BIOMARKERS:
            for band in BANDS:
                collected[strategy, family, band] = merge_rows(by_condition[family, band])

    # aame participants must occur in all tables
    ids = sorted(set.intersection(*(set(table.index) for table in collected.values())))
    if not ids:
        raise ValueError("No participant occurs in every strategy/condition table")

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

    finite_age = np.isfinite(ages.physical_age_years)
    ages["age_year"] = np.nan
    ages.loc[finite_age, "age_year"] = ages.loc[finite_age, "physical_age_years"].round().astype(int)

    arrays = {}
    for family in BIOMARKERS:
        for band in BANDS:
            arrays[family, band] = np.stack(
                [collected[s, family, band].reindex(ids).to_numpy(float) for s in STRATEGIES],
                axis=-1,
            )

    networks = np.array([parcel_network(parcel) for parcel in parcels])
    return arrays, ages, parcels, networks


# statistics
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


def paired_parcel_pvalues(template, individual):
    """two-sided paired t-test for each parcel"""
    difference = template - individual
    valid = np.isfinite(difference)
    n = valid.sum(axis=0)

    mean = np.divide(
        np.where(valid, difference, 0).sum(axis=0),
        n,
        out=np.full(difference.shape[1], np.nan),
        where=n > 0,
    )
    ss = np.where(valid, (difference - mean) ** 2, 0).sum(axis=0)
    sd = np.sqrt(np.divide(ss, n - 1, out=np.full_like(mean, np.nan), where=n > 1))

    ok = (n >= 2) & np.isfinite(sd) & (sd > 0)
    t = np.full_like(mean, np.nan)
    p = np.full_like(mean, np.nan)
    t[ok] = mean[ok] / (sd[ok] / np.sqrt(n[ok]))
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), n[ok] - 1)
    return p


def paired_scalar_pvalue(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan
    difference = x - y
    if not np.isfinite(difference).all() or np.std(difference, ddof=1) == 0:
        return np.nan
    return float(stats.ttest_rel(x, y).pvalue)


# parcel significance count
def parcel_count_summary(arrays, ages):
    rows = []

    for family in BIOMARKERS:
        for band in BANDS:
            raw = arrays[family, band]

            joint = np.isfinite(raw).all(axis=2)
            x = np.where(joint[:, :, None], raw, np.nan)

            for year in YEARS:
                age_mask = ages.age_year.eq(year).to_numpy()

                for label, strategy_index in (
                    ("Adult vs Individual", 2),
                    ("Pediatric vs Individual", 1),
                ):
                    p = paired_parcel_pvalues(
                        x[age_mask, :, strategy_index],
                        x[age_mask, :, 0],
                    )
                    q = bh_fdr(p)  # 100 parcel tests
                    rows.append({
                        "biomarker": family,
                        "band": band,
                        "age_year": year,
                        "comparison": label,
                        "n_significant": int(np.sum(q < 0.05)),
                    })

    counts = pd.DataFrame(rows)

    summary = (
        counts.groupby(["biomarker", "age_year", "comparison"], as_index=False)
        .agg(
            mean_count=("n_significant", "mean"),
            min_count=("n_significant", "min"),
            max_count=("n_significant", "max"),
        )
    )
    return counts, summary


def plot_parcel_figures(summary):
    for family in BIOMARKERS:
        fig, (ax_line, ax_heat) = plt.subplots(1, 2, figsize=(18, 7), layout="constrained")

        for label, marker, linestyle in (
            ("Adult vs Individual", "o", "-"),
            ("Pediatric vs Individual", "s", "--"),
        ):
            d = (
                summary[(summary.biomarker == family) & (summary.comparison == label)]
                .set_index("age_year")
                .reindex(YEARS)
            )
            ax_line.plot(YEARS, d.mean_count, marker=marker, linestyle=linestyle, label=label)
            ax_line.fill_between(YEARS, d.min_count, d.max_count, alpha=0.15)

        ax_line.set_xlabel("Age (years)")
        ax_line.set_ylabel("Mean significant-parcel count (q < 0.05)")
        ax_line.set_ylim(0, 100)
        ax_line.set_xticks(YEARS)
        ax_line.legend(frameon=False)
        ax_line.spines["top"].set_visible(False)
        ax_line.spines["right"].set_visible(False)

        heat_labels = ("Adult vs Individual", "Pediatric vs Individual")
        matrix = np.vstack([
            summary[(summary.biomarker == family) & (summary.comparison == label)]
            .set_index("age_year").reindex(YEARS).mean_count.to_numpy()
            for label in heat_labels
        ])

        image = ax_heat.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax_heat.set_xticks(range(len(YEARS)), YEARS)
        ax_heat.set_yticks([0, 1], ["Adult\nvs\nIndividual", "Pediatric\nvs\nIndividual"])
        ax_heat.set_xlabel("Age (years)")
        fig.colorbar(image, ax=ax_heat, label="Mean significant-parcel count")

        fig.suptitle(BIOMARKER_DISPLAY[family], fontweight="bold", fontsize=20)
        fig.savefig(
            OUTPUT_DIR / f"significant_parcels_{family}_lines_and_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# network absolute deviation analysis
def network_absolute_power(arrays, ages, parcel_networks):
    """Five-band mean Absolute-Power network deviations and network tests."""
    n_subjects = len(ages)
    # subject × band × network × template (Pediatric, Adult)
    errors = np.full((n_subjects, len(BANDS), len(NETWORKS), 2), np.nan)

    for bi, band in enumerate(BANDS):
        raw = arrays["AbsolutePower", band]
        joint = np.isfinite(raw).all(axis=2)
        x = np.where(joint[:, :, None], raw, np.nan)

        # subject × parcel × template
        abs_error = np.abs(x[:, :, 1:] - x[:, :, :1])

        for ni, network in enumerate(NETWORKS):
            parcel_mask = parcel_networks == network
            region = abs_error[:, parcel_mask, :]
            finite = np.isfinite(region)
            n = finite.sum(axis=1)
            errors[:, bi, ni, :] = np.divide(
                np.where(finite, region, 0).sum(axis=1),
                n,
                out=np.full((n_subjects, 2), np.nan),
                where=n > 0,
            )

    # mean across 5 bands within each subject
    complete = np.isfinite(errors).all(axis=1)
    five_band = np.where(complete, np.mean(errors, axis=1), np.nan)

    rows = []
    p_rows = []

    for year in YEARS:
        age_mask = ages.age_year.eq(year).to_numpy()
        p_values = []

        for ni, network in enumerate(NETWORKS):
            ped = five_band[age_mask, ni, 0]
            adult = five_band[age_mask, ni, 1]
            delta = ped - adult
            valid = np.isfinite(delta)

            rows.append({
                "age_year": year,
                "network": network,
                "n": int(valid.sum()),
                "delta": float(np.mean(delta[valid])) if valid.any() else np.nan,
            })

            p_values.append(paired_scalar_pvalue(ped, adult))

        q_values = bh_fdr(p_values)  # seven networks within this age
        for network, p, q in zip(NETWORKS, p_values, q_values):
            p_rows.append({
                "age_year": year,
                "network": network,
                "p": p,
                "q": q,
                "significant": bool(np.isfinite(q) and q < 0.05),
            })

    return pd.DataFrame(rows), pd.DataFrame(p_rows)


def plot_network(network, tests):
    matrix = (
        network.pivot(index="network", columns="age_year", values="delta")
        .reindex(index=NETWORKS, columns=YEARS)
        .to_numpy()
    )
    stars = (
        tests.pivot(index="network", columns="age_year", values="significant")
        .reindex(index=NETWORKS, columns=YEARS)
        .fillna(False)
        .to_numpy(bool)
    )

    limit = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(17, 8), layout="constrained")
    image = ax.imshow(matrix, cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(YEARS)), YEARS)
    ax.set_yticks(range(len(NETWORKS)), NETWORKS)
    ax.set_xlabel("Age (years)")
    ax.set_title(
        "Absolute Power Network deviations (mean across five frequency bands)",
        fontweight="bold",
    )

    for i in range(len(NETWORKS)):
        for j in range(len(YEARS)):
            if stars[i, j]:
                # Choose text contrast from cell magnitude.
                text_color = "white" if abs(matrix[i, j]) > 0.55 * limit else "black"
                ax.text(j, i, "*", ha="center", va="center", fontsize=22,
                        fontweight="bold", color=text_color)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "age-specific absolute deviation − adult absolute deviation"
    )
    fig.text(
        0.5,
        0.01,
        "* Paired network-error t-test: BH q < 0.05 across seven networks within each age.",
        ha="center",
    )

    fig.savefig(
        OUTPUT_DIR / "network_deviations_mean_AbsolutePower.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    arrays, ages, parcels, parcel_networks = load_data()

    counts, summary = parcel_count_summary(arrays, ages)
    counts.to_csv(OUTPUT_DIR / "significant_parcel_counts_by_band.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "significant_parcel_counts_five_band_summary.csv", index=False)
    plot_parcel_figures(summary)

    network, tests = network_absolute_power(arrays, ages, parcel_networks)
    network.to_csv(OUTPUT_DIR / "absolute_power_network_deviations.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "absolute_power_network_tests.csv", index=False)
    plot_network(network, tests)


if __name__ == "__main__":
    main()
