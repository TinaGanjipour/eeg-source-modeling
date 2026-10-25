#!/usr/bin/env python3
"""
Morph the fsaverage Schaefer-100/Yeo-7 cortical parcellation onto the age-specific pediatric anatomical templates.

Workflow:
1. Read the Schaefer-100/Yeo-7 annotation from fsaverage.
2. Exclude unknown/medial-wall labels, retaining 100 cortical ROIs (50 per hemisphere).
3. Identify the age-specific anatomical templates and verify the required FreeSurfer surfaces.
4. Morph the 100 cortical labels from fsaverage onto each pediatric template using MNE-Python.
5. Write the morphed labels as subject-specific FreeSurfer annotation files (.annot).
6. MNE assigns vertices not covered by the morphed ROIs to an additional "unknown" label in each hemisphere, where uncovered vertices exist.
7. Read back and validate the annotations, expecting 102 labels: 100 Schaefer ROIs + 2 unknown labels.
8. In the subsequent source reconstruction pipeline, the unknown labels are excluded so that the 100 Schaefer ROIs are used for regional analysis.

Additional quality control:
- Confirm 50 cortical ROIs per hemisphere.
- Verify that ROI names match the fsaverage parcellation.
- Check for empty morphed ROIs.
- Detect missing or invalid annotation files.
- Skip existing annotations that already pass validation.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import mne


DEFAULT_PARC = "Schaefer2018_100Parcels_7Networks_order"
SUBJECT_RE = re.compile(r"^ANTS(?P<age>\d+-(?:0|5))Years3T$")


def is_excluded_label(label: mne.Label) -> bool:
    name = label.name.lower()
    return "unknown" in name or "medial_wall" in name


def age_key(subject: str) -> tuple[int, int]:
    match = SUBJECT_RE.fullmatch(subject)
    if match is None:
        raise ValueError(subject)
    year, half = match.group("age").split("-")
    return int(year), int(half)


def read_annotation_pair(
    subject: str,
    subjects_dir: Path,
    lh_annot: Path,
    rh_annot: Path,
) -> list[mne.Label]:
    labels = []
    for annot in (lh_annot, rh_annot):
        labels.extend(
            mne.read_labels_from_annot(
                subject=subject,
                annot_fname=str(annot),
                subjects_dir=str(subjects_dir),
                surf_name="white",
                sort=True,
                verbose="ERROR",
            )
        )
    return labels


def validate_labels(
    subject: str,
    all_labels: list[mne.Label],
    expected_names: set[str],
) -> dict[str, int]:
    roi_labels = [label for label in all_labels if not is_excluded_label(label)]
    excluded = [label for label in all_labels if is_excluded_label(label)]

    if len(all_labels) != 102:
        raise RuntimeError(
            f"{subject}: expected 102 labels including unknown/Medial_Wall, "
            f"found {len(all_labels)}"
        )
    if len(roi_labels) != 100:
        raise RuntimeError(
            f"{subject}: expected 100 Schaefer ROIs, found {len(roi_labels)}"
        )
    if len(excluded) != 2:
        raise RuntimeError(
            f"{subject}: expected two excluded labels, found "
            f"{[label.name for label in excluded]}"
        )

    names = {label.name for label in roi_labels}
    if names != expected_names:
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        raise RuntimeError(
            f"{subject}: label names do not match fsaverage; "
            f"missing={missing}, extra={extra}"
        )

    empty = [label.name for label in roi_labels if len(label.vertices) == 0]
    if empty:
        raise RuntimeError(f"{subject}: empty morphed labels: {empty}")

    lh_count = sum(label.hemi == "lh" for label in roi_labels)
    rh_count = sum(label.hemi == "rh" for label in roi_labels)
    if (lh_count, rh_count) != (50, 50):
        raise RuntimeError(
            f"{subject}: expected 50 labels per hemisphere, "
            f"found lh={lh_count}, rh={rh_count}"
        )

    return {
        "all_labels": len(all_labels),
        "roi_labels": len(roi_labels),
        "excluded_labels": len(excluded),
        "lh_rois": lh_count,
        "rh_rois": rh_count,
    }


def annotation_paths(subject_dir: Path, parc: str) -> tuple[Path, Path]:
    label_dir = subject_dir / "label"
    return (
        label_dir / f"lh.{parc}.annot",
        label_dir / f"rh.{parc}.annot",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--parc", default=DEFAULT_PARC)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write annotations. Without this option, perform a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace incomplete or invalid existing annotation pairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects_dir = args.subjects_dir.expanduser().resolve()
    if not subjects_dir.is_dir():
        raise NotADirectoryError(subjects_dir)

    fsaverage_dir = subjects_dir / "fsaverage"
    if not fsaverage_dir.is_dir():
        raise NotADirectoryError(fsaverage_dir)

    subjects = sorted(
        (
            path.name
            for path in subjects_dir.iterdir()
            if path.is_dir() and SUBJECT_RE.fullmatch(path.name)
        ),
        key=age_key,
    )
    if len(subjects) != 26:
        raise RuntimeError(f"Expected 26 age templates, found {len(subjects)}")

    template_all = mne.read_labels_from_annot(
        subject="fsaverage",
        parc=args.parc,
        subjects_dir=str(subjects_dir),
        sort=True,
        verbose="ERROR",
    )
    template_labels = [
        label for label in template_all if not is_excluded_label(label)
    ]
    expected_names = {label.name for label in template_labels}

    if len(template_labels) != 100:
        raise RuntimeError(
            "Expected 100 fsaverage Schaefer labels after exclusions, "
            f"found {len(template_labels)}"
        )
    if (
        sum(label.hemi == "lh" for label in template_labels),
        sum(label.hemi == "rh" for label in template_labels),
    ) != (50, 50):
        raise RuntimeError("fsaverage does not contain 50 ROIs per hemisphere")

    planned: list[str] = []
    already_valid: list[str] = []
    problems: list[str] = []

    for subject in subjects:
        subject_dir = subjects_dir / subject
        required = (
            subject_dir / "surf" / "lh.white",
            subject_dir / "surf" / "rh.white",
            subject_dir / "surf" / "lh.sphere.reg",
            subject_dir / "surf" / "rh.sphere.reg",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            problems.append(f"{subject}: missing surfaces: {missing}")
            continue

        lh_annot, rh_annot = annotation_paths(subject_dir, args.parc)
        existing = (lh_annot.is_file(), rh_annot.is_file())
        if existing == (False, False):
            planned.append(subject)
            continue

        if existing == (True, True):
            try:
                labels = read_annotation_pair(
                    subject, subjects_dir, lh_annot, rh_annot
                )
                validate_labels(subject, labels, expected_names)
            except Exception as error:
                if args.overwrite:
                    planned.append(subject)
                else:
                    problems.append(
                        f"{subject}: existing annotations are invalid: {error}"
                    )
            else:
                already_valid.append(subject)
            continue

        if args.overwrite:
            planned.append(subject)
        else:
            problems.append(
                f"{subject}: only one hemisphere annotation exists"
            )

    print(f"Subjects directory:        {subjects_dir}")
    print(f"Parcellation:              {args.parc}")
    print(f"Subjects:                  {len(subjects)}")
    print(f"Already valid:             {len(already_valid)}")
    print(f"Labels planned:            {len(planned)}")
    print(f"Problems:                  {len(problems)}")

    if problems:
        print("\nProblems:")
        for problem in problems:
            print(problem)
        raise RuntimeError("No annotations were written because preflight failed")

    if not args.write:
        print("\nDRY RUN PASSED. No annotations were written.")
        print("Rerun the same command with --write.")
        return 0

    for index, subject in enumerate(planned, start=1):
        print(f"[{index}/{len(planned)}] Morphing labels: {subject}", flush=True)
        subject_dir = subjects_dir / subject
        lh_annot, rh_annot = annotation_paths(subject_dir, args.parc)
        lh_annot.parent.mkdir(parents=True, exist_ok=True)

        labels_individual = mne.morph_labels(
            template_labels,
            subject_to=subject,
            subject_from="fsaverage",
            subjects_dir=str(subjects_dir),
            surf_name="white",
            verbose="ERROR",
        )
        if len(labels_individual) != 100:
            raise RuntimeError(
                f"{subject}: morphing returned {len(labels_individual)} "
                "labels instead of 100"
            )
        empty = [
            label.name for label in labels_individual if len(label.vertices) == 0
        ]
        if empty:
            raise RuntimeError(f"{subject}: empty morphed labels: {empty}")

        mne.write_labels_to_annot(
            labels_individual,
            subject=subject,
            parc=args.parc,
            subjects_dir=str(subjects_dir),
            overwrite=args.overwrite,
            sort=True,
            verbose="ERROR",
        )

        readback = read_annotation_pair(
            subject, subjects_dir, lh_annot, rh_annot
        )
        qc = validate_labels(subject, readback, expected_names)
        print(
            f"  PASS: {qc['roi_labels']} ROIs; "
            f"lh={qc['lh_rois']}; rh={qc['rh_rois']}; "
            f"excluded={qc['excluded_labels']}"
        )

    print(
        f"\nPASS: all {len(subjects)} templates now have validated "
        "Schaefer-100 annotations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
References:
Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Lauri Parkkonen, Matti S. Hämäläinen, MNE software for processing MEG and EEG data, NeuroImage, Volume 86, 2014, Pages 446-460, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2013.10.027.
Alexander Schaefer, Ru Kong, Evan M Gordon, Timothy O Laumann, Xi-Nian Zuo, Avram J Holmes, Simon B Eickhoff, B T Thomas Yeo, Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI, Cerebral Cortex, Volume 28, Issue 9, September 2018, Pages 3095–3114, https://doi.org/10.1093/cercor/bhx179
Thomas Yeo BT, Krienen FM, Sepulcre J, et al. The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J Neurophysiol 106: 1125-1165, 2011. doi: 10.1152/jn.00338.2011.
"""
