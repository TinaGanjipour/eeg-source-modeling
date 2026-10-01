"""Coregistration (EEG-to-MRI transform)"""

from pathlib import Path
import mne


SUBJECT = ""
SUBJECTS_DIR = Path("")
EEG_FIF = Path("")
OUTPUT_TRANS = SUBJECTS_DIR / SUBJECT / "bem" / f"{SUBJECT}-trans.fif"


def main():
    # Read EEG
    raw = mne.io.read_raw_fif(
        EEG_FIF,
        preload=False,
        verbose=False,
    )

    # Initial alignment using MRI fiducials
    coreg = mne.coreg.Coregistration(
        raw.info,
        subject=SUBJECT,
        subjects_dir=SUBJECTS_DIR,
    )
    coreg.fit_fiducials()

    # Refine the transform using EEG electrodes only
    coreg.fit_icp(
        n_iterations=100,
        eeg_weight=1.0,
        hsp_weight=0.0,
        hpi_weight=0.0,
        lpa_weight=0.0,
        nasion_weight=0.0,
        rpa_weight=0.0,
    )

    OUTPUT_TRANS.parent.mkdir(parents=True, exist_ok=True)

    mne.write_trans(
        OUTPUT_TRANS,
        coreg.trans,
        overwrite=True,
    )

    raw.close()


if __name__ == "__main__":
    main()