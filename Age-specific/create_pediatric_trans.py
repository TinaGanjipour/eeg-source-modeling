"""Fit one common EEG digitization to every pediatric template.

For every ``ANTS{age}Years3T`` subject in ``--subjects-dir``:

1. read one EEG file without modifying it
2. copy its Info and remove EEG DigPoint IDs 48, 119, 126, and 127
3. fit fiducials with LPA=1, nasion=10, RPA=1
4. run EEG-only ICP for 100 iterations with every non-EEG weight set to zero
5. write the transform, numerical EEG-only QC, per-point distances, and one
   four-view alignment montage.
"""

from __future__ import annotations
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.io.constants import FIFF


SUBJECT_RE = re.compile(r"^ANTS(?P<age>\d+-(?:0|5))Years3T$")
EXCLUDED_EEG_DIGPOINT_IDS = (48, 119, 126, 127)
VIEWS = (("az000", 0.0), ("az090", 90.0), ("az180", 180.0), ("az270", 270.0))
ELEVATION = 90.0
THRESHOLD_MM = 4.0


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_info(path: Path) -> mne.Info:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".fif") or suffixes.endswith(".fif.gz"):
        return mne.io.read_info(str(path), verbose=False)
    raw = mne.io.read_raw(str(path), preload=False, verbose=False)
    return raw.info.copy()


def eeg_digpoints(info: mne.Info) -> list:
    return [
        point
        for point in (info.get("dig") or [])
        if int(point["kind"]) == int(FIFF.FIFFV_POINT_EEG)
    ]


def filtered_info(info: mne.Info) -> tuple[mne.Info, list[int]]:
    output = info.copy()
    removed: list[int] = []
    with output._unlock():
        kept = []
        for point in output.get("dig") or []:
            ident = int(point["ident"])
            if (
                int(point["kind"]) == int(FIFF.FIFFV_POINT_EEG)
                and ident in EXCLUDED_EEG_DIGPOINT_IDS
            ):
                removed.append(ident)
            else:
                kept.append(point)
        output["dig"] = kept
    remaining = {int(point["ident"]) for point in eeg_digpoints(output)}
    unexpected = remaining.intersection(EXCLUDED_EEG_DIGPOINT_IDS)
    if unexpected:
        raise RuntimeError(f"Excluded EEG DigPoint IDs remain: {sorted(unexpected)}")
    return output, sorted(removed)


def distances_mm(
    info: mne.Info, trans: mne.transforms.Transform, subject: str, subjects_dir: Path
) -> np.ndarray:
    return 1000.0 * np.asarray(
        mne.dig_mri_distances(
            info,
            trans,
            subject,
            subjects_dir=str(subjects_dir),
            dig_kinds="eeg",
            exclude_frontal=False,
            verbose=False,
        ),
        dtype=float,
    )


def fit_transform(
    info: mne.Info, subject: str, subjects_dir: Path
) -> mne.transforms.Transform:
    coreg = mne.coreg.Coregistration(
        info,
        subject=subject,
        subjects_dir=str(subjects_dir),
    )
    coreg.fit_fiducials(
        lpa_weight=1.0,
        nasion_weight=10.0,
        rpa_weight=1.0,
        verbose=True,
    )
    coreg.fit_icp(
        n_iterations=100,
        eeg_weight=1.0,
        hsp_weight=0.0,
        hpi_weight=0.0,
        lpa_weight=0.0,
        nasion_weight=0.0,
        rpa_weight=0.0,
        verbose=True,
    )
    return coreg.trans


def render_montage(
    info: mne.Info,
    trans_path: Path,
    subject: str,
    subjects_dir: Path,
    output_path: Path,
) -> None:
    kwargs = dict(
        info=info,
        trans=str(trans_path),
        subject=subject,
        subjects_dir=str(subjects_dir),
        surfaces="head",
        coord_frame="mri",
        meg=False,
        dig=True,
        eeg="original",
        mri_fiducials=False,
        show_axes=False,
        show=False,
    )
    try:
        figure = mne.viz.plot_alignment(**kwargs)
    except TypeError:
        kwargs.pop("show")
        figure = mne.viz.plot_alignment(**kwargs)

    image_paths = []
    try:
        plotter = getattr(figure, "plotter", None)
        if plotter is not None and hasattr(plotter, "set_background"):
            plotter.set_background((0.5, 0.5, 0.5))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for name, azimuth in VIEWS:
            mne.viz.set_3d_view(
                figure=figure,
                azimuth=azimuth,
                elevation=ELEVATION,
                distance="auto",
                focalpoint="auto",
            )
            image_path = output_path.with_name(f"{output_path.stem}_{name}.png")
            if hasattr(figure, "plotter") and hasattr(figure.plotter, "screenshot"):
                if hasattr(figure.plotter, "render"):
                    figure.plotter.render()
                figure.plotter.screenshot(str(image_path))
            elif hasattr(figure, "screenshot"):
                figure.screenshot(str(image_path))
            else:
                raise RuntimeError("The active MNE 3D figure has no screenshot method")
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise RuntimeError(f"Screenshot was not written: {image_path}")
            image_paths.append(image_path)
    finally:
        try:
            mne.viz.close_3d_figure(figure)
        except Exception:
            try:
                figure.close()
            except Exception:
                pass

    montage, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, image_path, (name, azimuth) in zip(axes, image_paths, VIEWS):
        axis.imshow(mpimg.imread(image_path))
        axis.set_title(f"{name}: az={azimuth:.0f}°, el={ELEVATION:.0f}°")
        axis.axis("off")
    montage.suptitle(subject)
    montage.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(montage)


def discover_subjects(subjects_dir: Path) -> list[Path]:
    subjects = [
        path
        for path in sorted(subjects_dir.iterdir())
        if path.is_dir() and SUBJECT_RE.fullmatch(path.name)
    ]
    if not subjects:
        raise RuntimeError(f"No ANTS age-template subjects found in {subjects_dir}")
    return subjects


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    subjects_dir = SUBJECTS_DIR.expanduser().resolve()
    eeg_path = EEG_PATH.expanduser().resolve()
    eeg_path = args.eeg_path.expanduser().resolve()
    if not subjects_dir.is_dir():
        raise NotADirectoryError(subjects_dir)
    if not eeg_path.is_file():
        raise FileNotFoundError(eeg_path)

    eeg_hash_before = sha256(eeg_path)
    original_info = read_info(eeg_path)
    info_fit, removed_ids = filtered_info(original_info)
    original_count = len(eeg_digpoints(original_info))
    filtered_points = eeg_digpoints(info_fit)
    filtered_count = len(filtered_points)
    point_ids = [int(point["ident"]) for point in filtered_points]
    subjects = discover_subjects(subjects_dir)

    print(f"Common EEG: {eeg_path}")
    print(f"Subjects: {len(subjects)}")
    print(f"Original EEG DigPoints: {original_count}")
    print(f"Removed IDs present in file: {removed_ids}")
    print(f"EEG DigPoints used for fitting/QC: {filtered_count}")
    if not RENDER:
        mne.viz.set_3d_backend(args.backend)

    qc_root = subjects_dir / "trans_qc_common_eeg_drop48_119_126_127"
    numerical_rows: list[dict] = []
    point_rows: list[dict] = []
    failures = []

    for index, subject_dir in enumerate(subjects, start=1):
        subject = subject_dir.name
        match = SUBJECT_RE.fullmatch(subject)
        assert match is not None
        age = match.group("age").replace("-", ".")
        bem_dir = subject_dir / "bem"
        trans_path = bem_dir / f"{subject}-trans.fif"
        print(f"[{index}/{len(subjects)}] {subject}", flush=True)
        try:
            required = (
                subject_dir / "mri" / "T1.mgz",
                subject_dir / "surf" / "lh.pial",
                bem_dir / "outer_skin.surf",
                bem_dir / f"{subject}-bem.fif",
                bem_dir / f"{subject}-sol.fif",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing required files: {missing}")

            if trans_path.exists() and not OVERWRITE:
                raise FileExistsError(
                    f"Transform already exists (use --overwrite to replace): {trans_path}"
                )
            trans = fit_transform(info_fit, subject, subjects_dir)
            mne.write_trans(str(trans_path), trans, overwrite=OVERWRITE, verbose=False)
            trans_readback = mne.read_trans(str(trans_path), return_all=False, verbose=False)
            distances = distances_mm(info_fit, trans_readback, subject, subjects_dir)
            if len(distances) != filtered_count:
                raise RuntimeError(
                    f"Distance count {len(distances)} != EEG DigPoint count {filtered_count}"
                )

            numerical_rows.append(
                {
                    "age": age,
                    "subject": subject,
                    "status": "PASS",
                    "eeg_subject": "sub-NDARBU532YXZ",
                    "original_eeg_digpoints": original_count,
                    "excluded_ids_requested": ";".join(map(str, EXCLUDED_EEG_DIGPOINT_IDS)),
                    "excluded_ids_found_and_removed": ";".join(map(str, removed_ids)),
                    "eeg_points_used": filtered_count,
                    "mean_mm": float(np.mean(distances)),
                    "median_mm": float(np.median(distances)),
                    "maximum_mm": float(np.max(distances)),
                    "points_over_4mm": int(np.sum(distances > THRESHOLD_MM)),
                    "percent_over_4mm": float(100.0 * np.mean(distances > THRESHOLD_MM)),
                    "trans_path": str(trans_path),
                }
            )
            point_rows.extend(
                {
                    "age": age,
                    "subject": subject,
                    "eeg_digpoint_id": point_id,
                    "distance_mm": float(distance),
                    "over_4mm": bool(distance > THRESHOLD_MM),
                }
                for point_id, distance in zip(point_ids, distances)
            )
            if not RENDER:
                render_montage(
                    info_fit,
                    trans_path,
                    subject,
                    subjects_dir,
                    qc_root / "montages" / f"{subject}_four_views.png",
                )
        except Exception as error:
            failures.append((subject, repr(error)))
            numerical_rows.append(
                {
                    "age": age,
                    "subject": subject,
                    "status": "FAIL",
                    "eeg_subject": "sub-NDARBU532YXZ",
                    "original_eeg_digpoints": original_count,
                    "excluded_ids_requested": ";".join(map(str, EXCLUDED_EEG_DIGPOINT_IDS)),
                    "excluded_ids_found_and_removed": ";".join(map(str, removed_ids)),
                    "eeg_points_used": filtered_count,
                    "mean_mm": "",
                    "median_mm": "",
                    "maximum_mm": "",
                    "points_over_4mm": "",
                    "percent_over_4mm": "",
                    "trans_path": str(trans_path),
                }
            )
            print(f"FAIL | {subject} | {error}", file=sys.stderr, flush=True)

        write_csv(
            qc_root / "trans_numerical_qc.csv",
            numerical_rows,
            list(numerical_rows[0]),
        )
        if point_rows:
            write_csv(
                qc_root / "trans_eeg_point_distances.csv",
                point_rows,
                list(point_rows[0]),
            )

    eeg_hash_after = sha256(eeg_path)
    if eeg_hash_after != eeg_hash_before:
        raise RuntimeError("The source EEG file changed during processing")
    manifest = {
        "subjects_dir": str(subjects_dir),
        "common_eeg_path": str(eeg_path),
        "common_eeg_sha256_before": eeg_hash_before,
        "common_eeg_sha256_after": eeg_hash_after,
        "source_eeg_unchanged": True,
        "excluded_eeg_digpoint_ids": list(EXCLUDED_EEG_DIGPOINT_IDS),
        "removed_ids_present_in_source": removed_ids,
        "fit_order": [
            "copy Info and exclude EEG DigPoint IDs 48,119,126,127",
            "fit_fiducials(lpa=1,nasion=10,rpa=1)",
            "EEG-only fit_icp(n_iterations=100; all non-EEG weights zero)",
            "numerical QC with dig_kinds=eeg",
            "four-view visual QC",
        ],
        "subjects": len(subjects),
        "passed": len(subjects) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "trans_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Transforms + numerical QC passed: {len(subjects) - len(failures)}/{len(subjects)}")
    print(f"QC root: {qc_root}")
    print("PASS: source EEG hash was unchanged")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
