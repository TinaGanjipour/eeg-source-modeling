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

T1 / anatomical model
│
├── Adult
│     fsaverage
│       → ico5 source space
│       → 3-layer BEM
│
├── Pediatric
│     nested tissue masks
│       → marching cubes
│       → Laplacian smoothing
│       → largest face-connected component + reindexing
│       → 5120-face decimation
│       → defect repair
│       → intersection correction
│       → 3-layer BEM
│
└── Individual
      T1 MRI
        → recon-all
        → mri_watershed
        → visual + numerical QC
        → ico5 source space
        → 3-layer BEM

EEG
│
GSN_HydroCel_129.sfp
  → common HydroCel sensor geometry
  → preprocessing/channel selection
  → 125 source-reconstruction channels
  → EEG-to-MRI transformation

Anatomy + EEG
        ↓
Forward solution
        ↓
Source reconstruction
        ↓
Schaefer-100 parcellation
        ↓
DFA / fE/I / absolute power / relative power
        ↓
Statistical analyses


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


Tina Ganjipour
MSc Bioinformatics and Systems Biology  
Vrije Universiteit Amsterdam / University of Amsterdam
