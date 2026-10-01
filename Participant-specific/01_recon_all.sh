# Each participant's T1-weighted MRI was processed using FreeSurfer's `recon-all` pipeline.
# This automated workflow performs intensity normalization, removal of non-brain tissue, anatomical registration, segmentation of subcortical structures and white matter, and reconstruction of the cortical surfaces.
# The white-matter and pial surfaces are generated for both hemispheres and corrected for topological defects, providing a participant-specific representation of cortical anatomy.
# These reconstructed anatomical surfaces were subsequently used as the basis for constructing the participant-specific EEG source model.

set -euo pipefail

SUBJECTS_DIR=""
SUBJECT=""
T1W=""
CPUS=""

recon-all \
    -sd "$SUBJECTS_DIR" \
    -s "$SUBJECT" \
    -i "$T1W" \
    -all \
    -openmp "$CPUS"

# References:
# Bruce Fischl. 2012. FreeSurfer. NeuroImage (2012). https://doi.org/10.1016/j.neuroimage.2012.01.021