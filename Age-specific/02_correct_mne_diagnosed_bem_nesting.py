# workflow inspired by O’Reilly et al., (2021)
from __future__ import annotations
import csv
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from create_pediatric_bem3_oreilly import (
    AGES,
    OUTPUT_SUBJECTS_DIR,
    attempt_mne_model_and_solution,
    fix_intersecting_surfaces,
    output_subject_name,
    write_json,
)


PAIR_PATHS = {
    "inner_skull_vs_outer_skull": ("inner_skull.surf", "outer_skull.surf"),
    "outer_skull_vs_outer_skin": ("outer_skull.surf", "outer_skin.surf"),
}


def back_up_surfaces_once(bem_dir: Path) -> Path:
    backup_dir = bem_dir / "before_mne_diagnosed_nesting_correction"
    if backup_dir.exists():
        raise FileExistsError(
            f"Correction backup already exists: {backup_dir}. Inspect the previous "
            "correction record instead of silently correcting twice."
        )
    backup_dir.mkdir()
    for name in ("inner_skull.surf", "outer_skull.surf", "outer_skin.surf"):
        shutil.copy2(bem_dir / name, backup_dir / name)
    return backup_dir


def correct_one(age: str) -> dict[str, object]:
    subject = output_subject_name(age)
    subject_dir = OUTPUT_SUBJECTS_DIR / subject
    bem_dir = subject_dir / "bem"
    status_path = bem_dir / "creation_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    with status_path.open() as stream:
        creation = json.load(stream)

    first_pair = creation.get("mne", {}).get("diagnosed_bem_pair")
    if creation.get("status") == "PASS":
        return {
            "age": age,
            "output_subject": subject,
            "status": "SKIPPED_ALREADY_PASS",
            "corrected_pairs": [],
            "error": "",
        }
    if first_pair not in PAIR_PATHS:
        return {
            "age": age,
            "output_subject": subject,
            "status": "SKIPPED_NO_MNE_NESTING_DIAGNOSIS",
            "corrected_pairs": [],
            "error": creation.get("mne", {}).get("error", ""),
        }

    backup_dir = back_up_surfaces_once(bem_dir)
    pair = first_pair
    corrected_pairs = []
    attempts = []

    for _ in range(len(PAIR_PATHS)):
        if pair not in PAIR_PATHS or pair in corrected_pairs:
            break
        inner_name, outer_name = PAIR_PATHS[pair]
        deltas = fix_intersecting_surfaces(
            bem_dir / inner_name, bem_dir / outer_name
        )
        moved = 0 if deltas is None else int((deltas > 0).sum())
        corrected_pairs.append(pair)
        attempts.append(
            {
                "diagnosed_pair": pair,
                "outer_surface": outer_name,
                "vertices_moved": moved,
            }
        )

        if moved == 0:
            result = {
                "model_status": "NOT_RUN",
                "solution_status": "NOT_RUN",
                "failure_stage": "oreilly_intersection_detector",
                "diagnosed_bem_pair": pair,
                "error": (
                    "MNE diagnosed this pair, but O'Reilly's detector proposed no "
                    "vertex movement. No arbitrary displacement was applied."
                ),
            }
            break

        result = attempt_mne_model_and_solution(
            subject, OUTPUT_SUBJECTS_DIR, subject
        )
        attempts[-1]["mne_after_correction"] = result
        if result["model_status"] == "PASS" and result["solution_status"] == "PASS":
            break
        pair = result.get("diagnosed_bem_pair")

    if result["model_status"] == "PASS" and result["solution_status"] == "PASS":
        status = "PASS"
    elif result.get("diagnosed_bem_pair") in corrected_pairs:
        status = "UNRESOLVED_REPEATED_NESTING_FAILURE"
    elif result.get("failure_stage") == "make_bem_solution":
        status = "SOLUTION_FAILURE_NOT_AN_INTERSECTION_DIAGNOSIS"
    elif result.get("diagnosed_bem_pair") is None:
        status = "MNE_FAILURE_NOT_CLASSIFIED_AS_BEM_NESTING"
    else:
        status = "UNRESOLVED"

    record = {
        "schema_version": 1,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "age": age,
        "output_subject": subject,
        "source_creation_status": str(status_path),
        "surface_backup": str(backup_dir),
        "corrected_pairs": corrected_pairs,
        "attempts": attempts,
        "final_mne": result,
        "status": status,
    }
    write_json(bem_dir / "nesting_correction_status.json", record)
    return record


def write_summary(rows: list[dict[str, object]]) -> None:
    path = OUTPUT_SUBJECTS_DIR / "bem_nesting_correction_summary.tsv"
    fields = (
        "age",
        "output_subject",
        "status",
        "corrected_pairs",
        "model_status",
        "solution_status",
        "error",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in rows:
            result = record.get("final_mne", {})
            writer.writerow(
                {
                    "age": record.get("age", ""),
                    "output_subject": record.get("output_subject", ""),
                    "status": record.get("status", ""),
                    "corrected_pairs": ";".join(record.get("corrected_pairs", [])),
                    "model_status": result.get("model_status", ""),
                    "solution_status": result.get("solution_status", ""),
                    "error": str(
                        result.get("error", record.get("error", ""))
                    ).replace("\n", " | "),
                }
            )


def main() -> int:
    rows = []
    for index, age in enumerate(AGES, start=1):
        print(f"[{index}/{len(AGES)}] {output_subject_name(age)}", flush=True)
        try:
            record = correct_one(age)
        except Exception:
            record = {
                "age": age,
                "output_subject": output_subject_name(age),
                "status": "CORRECTION_SCRIPT_FAILED",
                "corrected_pairs": [],
                "error": traceback.format_exc(),
            }
        rows.append(record)
        write_summary(rows)

    failures = [
        row
        for row in rows
        if row["status"]
        not in {"PASS", "SKIPPED_ALREADY_PASS"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
