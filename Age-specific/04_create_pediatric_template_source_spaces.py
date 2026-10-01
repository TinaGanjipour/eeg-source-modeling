"""Create and verify ico5 source spaces for pediatric template subjects"""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
import mne
import numpy as np

SUBJECTS_DIR = Path("")
AGE = None
OVERWRITE = False
SUBJECT_RE = re.compile(
    r"^ANTS(?P<age>\d+-(?:0|5))Years3T$"
)

def age_key(age: str) -> tuple[int, int]:
    year, half = age.split("-")
    return int(year), int(half)


def discover_subjects(subjects_dir: Path, requested_age: str | None) -> list[tuple[str, str]]:
    matches: dict[str, list[str]] = {}
    for path in subjects_dir.iterdir():
        if not path.is_dir():
            continue
        match = SUBJECT_RE.fullmatch(path.name)
        if match is None:
            continue
        age = match.group("age")
        if requested_age is not None and age != requested_age:
            continue
        matches.setdefault(age, []).append(path.name)

    if requested_age is not None and requested_age not in matches:
        raise RuntimeError(
            f"No template directory was found for age {requested_age} in {subjects_dir}"
        )
    ambiguous = {age: names for age, names in matches.items() if len(names) != 1}
    if ambiguous:
        raise RuntimeError(f"Ambiguous template directories: {ambiguous}")
    if not matches:
        raise RuntimeError(f"No pediatric template subjects found in {subjects_dir}")
    return [
        (age, matches[age][0])
        for age in sorted(matches, key=age_key)
    ]


def validate_src(src: mne.SourceSpaces, expected_spacing: str = "ico5") -> dict:
    if len(src) != 2:
        raise RuntimeError(f"Expected two hemispheres; found {len(src)}")
    hemispheres = []
    for index, hemi in enumerate(src):
        rr = np.asarray(hemi["rr"])
        vertno = np.asarray(hemi["vertno"], dtype=int)
        inuse = np.asarray(hemi["inuse"], dtype=int)
        if not np.isfinite(rr).all():
            raise RuntimeError(f"Hemisphere {index} contains non-finite coordinates")
        if int(hemi["nuse"]) != len(vertno):
            raise RuntimeError(f"Hemisphere {index}: nuse and vertno disagree")
        if not np.array_equal(np.flatnonzero(inuse), vertno):
            raise RuntimeError(f"Hemisphere {index}: inuse and vertno disagree")
        if str(hemi.get("type", "surf")) != "surf":
            raise RuntimeError(f"Hemisphere {index} is not a surface source space")
        hemispheres.append(
            {
                "hemisphere": "lh" if index == 0 else "rh",
                "nuse": int(hemi["nuse"]),
                "np": int(hemi["np"]),
                "finite_coordinates": True,
            }
        )
    return {
        "spacing": expected_spacing,
        "hemispheres": hemispheres,
        "total_sources": int(sum(item["nuse"] for item in hemispheres)),
    }


def create_one(subjects_dir: Path, age: str, subject: str, overwrite: bool) -> dict:
    subject_dir = subjects_dir / subject
    canonical_subject = f"ANTS{age}Years3T"
    bem_dir = subject_dir / "bem"
    out_src = bem_dir / f"{canonical_subject}-src.fif"
    required = (
        subject_dir / "surf" / "lh.white",
        subject_dir / "surf" / "rh.white",
        subject_dir / "surf" / "lh.sphere",
        subject_dir / "surf" / "rh.sphere",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source-space inputs: {missing}")

    if out_src.is_file() and not overwrite:
        src = mne.read_source_spaces(str(out_src), verbose=False)
        qc = validate_src(src)
        status = "SKIPPED_EXISTING_VALID"
    else:
        src = mne.setup_source_space(
            subject=subject,
            spacing="ico5",
            subjects_dir=str(subjects_dir),
            n_jobs=1,
            verbose=True,
        )
        mne.write_source_spaces(
            str(out_src),
            src,
            overwrite=True,
        )
        src = mne.read_source_spaces(str(out_src), verbose=False)
        qc = validate_src(src)
        status = "CREATED_AND_VERIFIED"

    if not out_src.is_file() or out_src.stat().st_size == 0:
        raise RuntimeError(f"Source-space output is absent or empty: {out_src}")
    return {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "age": age,
        "subject_directory": subject,
        "canonical_subject": canonical_subject,
        "source_space": str(out_src),
        "status": status,
        **qc,
    }


def main() -> int:
    subjects_dir = SUBJECTS_DIR.expanduser().resolve()
    if not subjects_dir.is_dir():
        raise NotADirectoryError(subjects_dir)
    if AGE is not None and not re.fullmatch(r"\d+-(?:0|5)", AGE):
        raise ValueError(f"Invalid age: {AGE}")

    qc_dir = subjects_dir / "source_space_qc"
    qc_dir.mkdir(exist_ok=True)
    failures = []
    subjects = discover_subjects(subjects_dir, AGE)
    for index, (age, subject) in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] AGE={age} SUBJECT={subject}", flush=True)
        try:
            record = create_one(subjects_dir, age, subject, OVERWRITE)
            qc_path = qc_dir / f"ANTS{age}Years3T-src-qc.json"
            qc_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(f"SOURCE_SPACE_STATUS={record['status']}")
            print(f"SOURCE_SPACE_FILE={record['source_space']}")
            print(f"LH_NUSE={record['hemispheres'][0]['nuse']}")
            print(f"RH_NUSE={record['hemispheres'][1]['nuse']}")
            print(f"TOTAL_SOURCES={record['total_sources']}")
            print("FINAL_EXIT_STATUS=0", flush=True)
        except Exception as error:
            failures.append((age, repr(error)))
            print(f"SOURCE_SPACE_STATUS=FAIL")
            print(f"ERROR={error!r}")
            print("FINAL_EXIT_STATUS=1", flush=True)
    if failures:
        print("FAILURES:")
        for age, error in failures:
            print(f"  {age}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
