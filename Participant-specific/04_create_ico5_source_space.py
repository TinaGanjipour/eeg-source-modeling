"""Create participant-specific ico-5 cortical source space"""

from pathlib import Path
import mne

SUBJECT = ""
SUBJECTS_DIR = Path("")
OUTPUT_SRC = (
    SUBJECTS_DIR
    / SUBJECT
    / "bem"
    / f"{SUBJECT}-ico5-src.fif"
)

def main():
    src = mne.setup_source_space(
        subject=SUBJECT,
        spacing="ico5",
        subjects_dir=SUBJECTS_DIR,
        add_dist=False,
    )

    OUTPUT_SRC.parent.mkdir(parents=True, exist_ok=True)

    mne.write_source_spaces(
        OUTPUT_SRC,
        src,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
