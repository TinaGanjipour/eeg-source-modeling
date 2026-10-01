# EEG Source Modeling

Analysis scripts associated with the MSc thesis:
**Age-specific Pediatric Templates Improve EEG Biomarker Estimates Across Developmental Populations**


## Anatomical modelling workflow

Three anatomical modelling strategies were used:
1. Adult template (`fsaverage`)
2. Pediatric age-specific templates from Richards et al.'s database
3. Participant-specific MRI models

All approaches ultimately produced the anatomical inputs required for EEG
forward modelling: three-layer BEM surfaces (Conductivities were set as brain: 0.3 S/m, skull: 0.006 S/m, scalp: 0.3 S/m), cortical source space, and
head-to-MRI transformation.

### Adult template

The FreeSurfer `fsaverage` anatomy was used as the general adult template.
An ico-5 cortical source space and three-layer BEM model were constructed
using MNE-Python.

### Pediatric age-specific template

For each pediatric template, nested tissue masks were converted into
BEM surfaces using marching-cubes extraction. Surfaces were Laplacian
smoothed, after which the largest face-connected mesh component was
selected and vertex/face indices were rebuilt. Each surface was
spherically decimated to 5,120 triangles, repaired for mesh defects,
and checked/corrected for intersections between the nested BEM layers.

### Participant-specific MRI models

Participant T1-weighted MRIs were processed with FreeSurfer `recon-all`.
Three-layer BEM surfaces were subsequently generated using
`mri_watershed`. BEM surfaces underwent visual and numerical quality
control before being used for source-space construction and EEG-to-MRI
registration.

## EEG electrode information

The Healthy Brain Network EEG recordings were acquired using the 129-channel EGI HydroCel Geodesic Sensor Net. Participant-specific electrode coordinates and individually measured fiducials were not available in the released dataset used in this study. Therefore, the provided `GSN_HydroCel_129.sfp` file was used to define a common HydroCel sensor geometry across participants.

For source reconstruction, the EEG sensor information was processed as follows:

```text
GSN_HydroCel_129.sfp
        ↓
Common HydroCel sensor geometry
        ↓
EEG preprocessing and channel selection
        ↓
Removal of E48, E119, E126, and E127
        ↓
125-channel source-reconstruction montage
        ↓
EEG-to-MRI transformation
        ↓
Forward-model computation
```

The same sensor geometry was used for the participant-specific MRI, pediatric age-specific template, and adult `fsaverage` anatomical modelling strategies, while the EEG-to-MRI transformation was estimated separately for each anatomical model.

## Source reconstruction and biomarker analysis

The anatomical models and EEG sensor information were combined to compute the forward solution for each reconstruction strategy. The forward models were then used for EEG source reconstruction, followed by cortical parcellation and neurophysiological biomarker extraction.

```text
Anatomical model + EEG sensor information
        ↓
Forward solution
        ↓
EEG source reconstruction
        ↓
Schaefer2018 100-parcel cortical parcellation
        ↓
Parcel-level biomarker extraction
        ↓
DFA / fE/I / absolute power / relative power
        ↓
Statistical comparison of reconstruction strategies
```

The same downstream source-reconstruction and biomarker-analysis workflow was applied to the participant-specific MRI, pediatric age-specific template, and adult `fsaverage` models.


Tina Ganjipour
MSc Bioinformatics and Systems Biology  
Vrije Universiteit Amsterdam / University of Amsterdam
