# Following anatomical reconstruction, participant-specific boundary-element model (BEM) surfaces were generated from the T1-weighted MRI using FreeSurfer's `mri_watershed` algorithm.
# The watershed procedure separates brain and non-brain tissues based on image intensity and subsequently fits deformable surfaces to the estimated anatomical boundaries.
# Using the `-surf` option, three nested surfaces representing the inner skull, outer skull, and outer skin were generated for subsequent EEG forward-model construction.

set -euo pipefail

WS_PATH=""
WATERSHED_DIR=""
SUBJECT=""
T1_MGZ=""
WS_MGZ=""

mkdir -p "$WATERSHED_DIR"

mri_watershed \
    "$WS_PATH" \
    -h 20 \
    -useSRAS \
    -surf "$WATERSHED_DIR/$SUBJECT" \
    -shk_br_surf 3 "$WATERSHED_DIR/${SUBJECT}_brain" \
    "$T1_MGZ" \
    "$WS_MGZ"

# References:
# Ségonne, F et al. “A hybrid approach to the skull stripping problem in MRI.” NeuroImage vol. 22,3 (2004): 1060-75. doi:10.1016/j.neuroimage.2004.03.032
# Bruce Fischl. 2012. FreeSurfer. NeuroImage (2012). https://doi.org/10.1016/j.neuroimage.2012.01.021