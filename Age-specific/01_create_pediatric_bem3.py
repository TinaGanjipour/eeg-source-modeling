#!/usr/bin/env python3
"""
Pediatric age-specific BEM surface reconstruction pipeline.

The BEM surface extraction, geometric processing, mesh-repair, and intersection-correction procedures were adapted from:
Christian O'Reilly, Eric Larson, John E. Richards, Mayada Elsabbagh (2021)
Structural templates for imaging EEG cortical sources in infants. NeuroImage, 227, 117682.
https://doi.org/10.1016/j.neuroimage.2020.117682.
https://github.com/christian-oreilly/infant_template_paper

Anatomical inputs:
Neurodevelopmental MRI Database (Richards et al., 2016).

In this study, the templates from the children and adolescent’s sets were used, covering ages 6.0 – 17.0 years, at one-year intervals.

Workflow (adapted from O'Reilly et al., 2021):
1. Disconnected-artifact cleanup
2. Convert the segmented average MRI volume into cumulative binary masks
3. Extract the BEM surface using marching cubes (isovalue = 0.5)
4. Convert voxel coordinates into FreeSurfer surface RAS
5. Correct surface-normal orientation
6. Apply Laplacian smoothing (λ = 0.8, 15 iterations)
7. Perform mesh defect detection, repair, and validation
8. Largest-component selection
9. Apply spherical decimation to 5120 triangles per surface
10. Repeat mesh defect detection, repair, and validation
11. Perform pial-to-inner-skull intersection correction
12. Construct the three-layer MNE BEM model and solution

Geometric and topological quality control:
- Degenerate triangle detection and removal
- Detection of vertices incident to fewer than three triangles
- Removal of isolated vertices
- Removal of dangling faces and associated vertices
- Local triangle reconstruction around two-face vertices
- Mesh vertex reindexing after deletion
- Hole repair using PyMeshFix
- Surface-normal correction after hole repair
- Solid-angle-based surface completeness checks
- Watertightness checks
- Final validation of mesh integrity
"""
from __future__ import annotations
import csv
import json
import os
import re
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence
import cc3d
import mne
import nibabel as nib
import numpy as np
import pandas as pd
import pymeshfix
import trimesh
from mne.bem import _get_solids
from mne.surface import _triangle_neighbors, decimate_surface
from nibabel.freesurfer.io import read_geometry, write_geometry
from skimage import measure

SUBJECTS_DIR = Path("")
BEM_VOLUME_DIR = SUBJECTS_DIR # The segmented volume obtained from John E. Richard et al., 2016.
OUTPUT_SUBJECTS_DIR = Path("")

ALL_AGES = tuple(f"{year}-{half}" for year in range(6, 17) for half in (1, 0))
AGES: Sequence[str] = ALL_AGES
OVERWRITE = False

"""
Sphere decimation of surfaces to 5120 triangles was intended to make the comparison between the three anatomical modelling strategies methodologically consistent and interpretable.
5120 triangles correspond to an icosahedral subdivision grade of 4 (ico=4) in MNE-Python.
In this study, we compared three anatomical strategies:
1. Participant-specific MRI
2. Pediatric age-specific templates
3. Adult template (fsaverage)
By using the same target number of triangles for all three strategies, we aimed to standardize the BEM mesh resolution, so that differences in EEG source reconstruction would primarily reflect differences in anatomical modelling rather than differences in mesh density.
"""
TARGET_TRIANGLES = 5120

"""
These are the brain, skull, and scalp conductivities, respectively, in S/m.
The same conductivity values were applied throughout the developmental range.
"""
CONDUCTIVITY = (0.3, 0.006, 0.3)

"""
Pial-to-inner-skull intersection correction:
MAX_CORRECTION_DISTANCE_MM determines which geometric inconsistencies are small enough to attempt to correct.
MOVE_MARGIN_MM determines how much additional displacement is applied beyond the detected intersection distance.
"""
MOVE_MARGIN_MM = 0.5
MAX_CORRECTION_DISTANCE_MM = 5.0

"""The segmented BEM3 NIfTI contains three nested tissue compartments and background.
Every voxel is labeled.
Instead of independently extracting each exact label as a separate tissue class, they are constructed cumulative masks (O'Reilly et al. 2021).

- outer_skin: labels 1, 2, and 3
- outer_skull: labels 1 and 2
- inner_skull: label 1
​"""
SURFACE_LABELS = {
    "outer_skin": (1, 2, 3),
    "outer_skull": (1, 2),
    "inner_skull": (1,),
}

"""During BEM model construction, MNE-Python checks whether the three anatomical boundary surfaces are correctly nested:
the inner skull must lie entirely within the outer skull, and the outer skull within the scalp surface.
These checks are essential because valid, non-intersecting BEM boundaries are required to compute the BEM solution, which is subsequently used for EEG forward modelling.
These nesting violations are identified through MNE-generated error messages and the affected surface pairs are recorded by classify_mne_nesting_failure function for further correction.
"""
NESTING_PATTERNS = {
    "inner_skull_vs_outer_skull": re.compile(
        r"Surface inner skull is not (?:completely )?inside surface outer skull",
        re.IGNORECASE,
    ),
    "outer_skull_vs_outer_skin": re.compile(
        r"Surface outer skull is not (?:completely )?inside surface outer skin",
        re.IGNORECASE,
    ),
}

def output_subject_name(age: str) -> str:
    return f"ANTS{age}Years3T"

"""
The MRI inputs:
1. The segmented average MRI (AVG{age}Years3T_segmented_BEM3.nii.gz). The three BEM boundaries are extracted from these.
2. The FreeSurfer anatomical files, used for coordinate metadata and cortical surface geometry.
"""
def input_paths(age: str) -> dict[str, Path]:
    source_subject = f"ANTS{age}Years3T"
    subject_dir = SUBJECTS_DIR / source_subject
    return {
        "source_subject": Path(source_subject),
        "subject_dir": subject_dir,
        "bem_volume": BEM_VOLUME_DIR / f"AVG{age}Years3T_segmented_BEM3.nii.gz",
        "lh_white": subject_dir / "surf" / "lh.white",
        "lh_pial": subject_dir / "surf" / "lh.pial",
        "rh_pial": subject_dir / "surf" / "rh.pial",
        "t1": subject_dir / "mri" / "T1.mgz",
    }

def preflight() -> None:
    missing = []
    for age in AGES:
        paths = input_paths(age)
        for key in ("bem_volume", "lh_white", "lh_pial", "rh_pial", "t1"):
            if not paths[key].is_file():
                missing.append(paths[key])
    if missing:
        raise FileNotFoundError(
            "Missing inputs:\n" + "\n".join(f"  {path}" for path in missing)
        )

"""
Small disconnected pieces in the segmentation could generate unwanted surfaces during marching cubes.
Connected-component analysis is applied to the non-background voxels to identify disconnected artifacts (O'Reilly et al. 2021).
The procedure retains the background and main foreground components and sets the remaining components to zero.
"""
def correct_line_artefact(volume: np.ndarray) -> None:
    components = cc3d.connected_components((volume != 0).astype(int)) # Connected-component analysis: find spatially connected regions in the segmented MRI.
    label_id, count = np.unique(components, return_counts=True)
    id_zero, id_non_zero = label_id[count > 500_000] # Exactly two connected-component labels exceed 500,000 voxels
    artefact = np.where((components != id_zero) & (components != id_non_zero))
    volume[artefact] = 0 # then we keep the large connected components identified by its size threshold and set the remaining components to zero.

# Solid-angle-based surface completeness check: evaluating whether the mesh forms a consistently oriented, closed enclosure around the selected reference point.
def surface_is_complete(vertices: np.ndarray, faces: np.ndarray) -> bool:
    center = vertices.mean(axis=0)
    total_angle = _get_solids(vertices[faces], center[np.newaxis, :])[0]
    proportion = total_angle / (2 * np.pi)
    return bool(np.abs(proportion - 1.0) < 1e-5)

# Degenerate face = zero-area and near-zero-area triangles (O'Reilly et al., 2021)
def nondegenerate_face_mask(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices, faces)
    if hasattr(mesh, "nondegenerate_faces"):
        return np.asarray(mesh.nondegenerate_faces(), dtype=bool)
    return np.asarray(mesh.remove_degenerate_faces(), dtype=bool)

def has_degenerate_faces(vertices: np.ndarray, faces: np.ndarray) -> bool:
    return not bool(np.all(nondegenerate_face_mask(vertices, faces)))

def remove_degenerate_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices, faces)
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    else:
        mesh.remove_degenerate_faces()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)

# Identify potentially defective vertices by counting the number of triangular faces incident to each vertex (O'Reilly et al., 2021).
def get_topological_defects(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[list[int], list[int], list[int]]:
    neighbors = _triangle_neighbors(faces, len(vertices))
    zero, one, two = [], [], []
    for vertex_id, triangles in enumerate(neighbors):
        if len(triangles) == 0:
            zero.append(vertex_id) # 0 faces: isolated vertex, not connected to any triangle
        elif len(triangles) == 1:
            one.append(vertex_id) # 1 face: vertex belonging to only one triangle
        elif len(triangles) == 2:
            two.append(vertex_id) # 2 faces: vertex belonging to only two triangles
    return zero, one, two


def has_topological_defects(vertices: np.ndarray, faces: np.ndarray) -> bool:
    return any(len(group) for group in get_topological_defects(vertices, faces))


"""
The two triangles sharing the defective vertex are identified and scheduled for removal (O'Reilly et al., 2021).
Their remaining unique vertices are used to construct a single replacement triangle, eliminating the underconnected vertex.
"""
def correction_two_neighboring_triangles(
    vertices: np.ndarray, faces: np.ndarray, faulty_vertices: Sequence[int]
) -> tuple[np.ndarray, list[np.ndarray]]:
    faces_to_remove = []
    new_faces = []
    for vertex_id in faulty_vertices:
        face_ids = np.where(faces == vertex_id)[0]
        faces_to_remove.extend(face_ids)
        face_1, face_2 = faces[face_ids]
        new_face = np.unique(np.concatenate((face_1, face_2)))
        new_face = np.delete(new_face, np.where(new_face == vertex_id))
        if len(new_face) != 3:
            raise RuntimeError("The two triangles do not share one common edge.")
        new_det = np.linalg.det(vertices[new_face])
        if not new_det:
            raise RuntimeError("Topology repair produced three collinear points.")
        if np.sign(np.linalg.det(vertices[face_1])) == np.sign(new_det):
            new_face = new_face[[1, 0, 2]]
        new_faces.append(new_face)
    return np.asarray(faces_to_remove, dtype=int), new_faces

"""
In triangular meshes, the faces array does not directly store coordinates.
Instead, each face stores three indices pointing to entries in the vertices array.
If you remove a vertex, the remaining vertices must be renumbered. Hence, we update mesh vertex indices after defective vertices have been removed (O'Reilly et al., 2021).
"""
def reindex_vertices(
    vertices: np.ndarray, faces: np.ndarray, vertices_to_remove: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    decrement = np.cumsum(
        np.isin(np.arange(vertices.shape[0]), vertices_to_remove).astype(int)
    )
    vertices = np.delete(vertices, vertices_to_remove, axis=0)
    faces = faces - decrement[faces]
    return vertices, faces

"""
fix_topological_defects() handles the three categories returned by get_topological_defects().
For each category, the intended behavior (detected defect → intended correction) is:
Zero-face vertex → Remove the isolated vertex
One-face vertex → Remove the affected face and vertex
Two-face vertex → Replace the two neighboring triangles with one triangle, then remove the defective vertex
(O'Reilly et al., 2021)
"""
def fix_topological_defects(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    zero, one, two = get_topological_defects(vertices, faces)
    faces_to_remove = []
    if one:
        faces_to_remove.extend(np.where(faces == one)[0].tolist())
    if two:
        remove, faces_to_add = correction_two_neighboring_triangles(
            vertices, faces, two
        )
        faces_to_remove.extend(remove)
        faces = np.concatenate(
            (
                np.delete(
                    faces,
                    np.asarray(faces_to_remove, dtype=int),
                    axis=0,
                ),
                faces_to_add,
            )
        )
    vertices_to_remove = np.concatenate((zero, one, two)).astype(int)
    if len(vertices_to_remove):
        vertices, faces = reindex_vertices(vertices, faces, vertices_to_remove)
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)

"""
If a surface contains an opening caused by missing triangles, it may be possible to reconstruct the missing region and produce a closed surface.
MeshFix attempts to correct defects such as holes and reconstruct a valid triangular surface.
(O'Reilly et al., 2021)
"""
def repair_holes(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    meshfix = pymeshfix.MeshFix(vertices, faces)
    meshfix.repair()
    repaired_vertices = meshfix.v if hasattr(meshfix, "v") else meshfix.points
    repaired_faces = meshfix.f if hasattr(meshfix, "f") else meshfix.faces
    mesh = trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces) # convert the repaired result back to a Trimesh object
    trimesh.repair.fix_normals(mesh, multibody=False) # correct normal orientation
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)


def check_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    if not surface_is_complete(vertices, faces):
        raise RuntimeError("Surface is incomplete by the solid-angle check.")
    if has_topological_defects(vertices, faces):
        raise RuntimeError("Surface has vertices with fewer than three triangles.")
    if has_degenerate_faces(vertices, faces):
        raise RuntimeError("Surface has degenerate faces.")
    if not trimesh.Trimesh(vertices, faces).is_watertight:
        raise RuntimeError("Surface is not watertight.") # watertight = surface behaving like a closed subject (every mesh edge belongs to exactly two faces)


def fix_all_defects(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if has_degenerate_faces(vertices, faces):
        vertices, faces = remove_degenerate_faces(vertices, faces)
        if has_degenerate_faces(vertices, faces):
            raise RuntimeError("Degenerate-face removal did not converge.")
    if has_topological_defects(vertices, faces):
        vertices, faces = fix_topological_defects(vertices, faces)
        if has_degenerate_faces(vertices, faces):
            vertices, faces = remove_degenerate_faces(vertices, faces)
        if has_topological_defects(vertices, faces):
            raise RuntimeError("Topology repair did not converge.")
    if not surface_is_complete(vertices, faces) or not trimesh.Trimesh(
        vertices, faces
    ).is_watertight:
        vertices, faces = repair_holes(vertices, faces)
    check_mesh(vertices, faces)
    return vertices, faces
"""
Separate the triangular mesh into connected components and retain the largest component by triangle count (Dogdas et al., 2005).
It removes smaller disconnected mesh fragments that may remain after surface extraction and repair.
If only one connected component exists, the original mesh is returned unchanged.
The selected main component is then passed to spherical surface decimation.
"""
def select_largest_component(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices, faces)
    components = list(mesh.split(only_watertight=False))
    if not components:
        raise RuntimeError("No mesh component was found before decimation.")

    if len(components) == 1:
        return (
            np.asarray(vertices, dtype=float),
            np.asarray(faces, dtype=np.int32),
        )

    main = max(components, key=lambda component: len(component.faces))
    return np.asarray(main.vertices), np.asarray(main.faces, dtype=np.int32)

"""
Marching cubes takes a three-dimensional binary mask and extracts a polygonal boundary between the inside and outside.
The resulting mesh contains vertices (3D coordinates of mesh points) and faces (triangles connecting the vertices).
The level=0.5 parameter determines the boundary between values 0 (outside) and 1 (inside).
(O'Reilly et al. 2021)
"""
def extract_surface(
    mask: np.ndarray,
    affine: np.ndarray,
    volume_info: dict,
    destination: Path,
) -> None:
    vertices, faces = measure.marching_cubes(
        mask.astype(float),
        level=0.5,
        spacing=(1, 1, 1),
        allow_degenerate=False,
        method="lewiner",
    )[:2]
    """
    Marching-cubes output is in voxel coordinates, and the new BEM surfaces are transformed to share the appropriate coordinate frame with the FreeSurfer cortical surfaces using cras.
    The cras metadata comes from the FreeSurfer lh.white surface:
    """
    vertices = (
        vertices @ affine[:3, :3]
        + affine[:3, 3]
        - np.asarray(volume_info["cras"])
    )

    """
    Fixing surface orientation (O'Reilly et al. 2021):
    The orientation matters because MNE expects surfaces with consistent outward orientation.
    Incorrect normals can cause problems when constructing the BEM.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh.repair.fix_normals(mesh, multibody=False)

    """
    To reduce small irregularities introduced by the voxel-based segmentation and surface extraction, Laplacian smoothing is applied (O'Reilly et al. 2021).
    The parameter lamb=0.8 controls the smoothing strength, and iterations=15 specifies how many smoothing passes are performed.
    """
    mesh = trimesh.smoothing.filter_laplacian(
        deepcopy(mesh), lamb=0.8, iterations=15, volume_constraint=True
    )
    vertices, faces = fix_all_defects(mesh.vertices, mesh.faces)
    write_geometry(
        str(destination.with_name(destination.stem + "_large.surf")),
        vertices,
        faces,
        volume_info=volume_info,
    )

    vertices, faces = select_largest_component(vertices, faces)
    
    # Spherical decimation (O'Reilly et al. 2021)
    vertices, faces = decimate_surface(
        vertices, faces, n_triangles=TARGET_TRIANGLES, method="sphere"
    )
    vertices, faces = fix_all_defects(vertices, faces)
    if len(faces) != TARGET_TRIANGLES:
        raise RuntimeError(
            f"{destination.name}: expected {TARGET_TRIANGLES} faces after "
            f"required repairs, found {len(faces)}."
        )
    write_geometry(
        str(destination), vertices, faces, volume_info=volume_info
    )


def process_bem_volume(bem_volume: Path, lh_white: Path, bem_dir: Path) -> None:
    image = nib.load(str(bem_volume))
    volume_info = read_geometry(str(lh_white), read_metadata=True)[2]
    if "cras" not in volume_info:
        raise RuntimeError(f"No cras metadata in {lh_white}")

    for surface_name, labels in SURFACE_LABELS.items():
        data = deepcopy(image.get_fdata())
        correct_line_artefact(data)
        mask = np.isin(data, labels)
        extract_surface(
            mask,
            image.affine,
            volume_info,
            bem_dir / f"{surface_name}.surf",
        )

"""
Pial-to-inner-skull intersection correction:
Age-specific templates contain two independently derived anatomical structures:
1. Pial surface: The outer boundary of the cerebral cortex, provided by the FreeSurfer-processed anatomical template.
2. Inner-skull surface: The boundary surrounding the brain compartment, which we extracted from the segmented MRI using marching cubes.
Ideally, the entire pial surface must be enclosed by the inner-skull boundary.
However, because these surfaces originate from different processing procedures, they can occasionally intersect or cross one another.
These anatomical intersections that could cause problems during source modelling.
To ensure that the cortical pial surfaces lie inside the inner-skull BEM boundary, the affected inner-skull vertices were moved outward (O'Reilly et al. 2021).
"""
def intersection_deltas(
    inner_surface: Path,
    outer_surface: Path,
    move_margin_mm: float = MOVE_MARGIN_MM,
) -> np.ndarray:
    inner_mesh = trimesh.Trimesh(
        *read_geometry(str(inner_surface), read_metadata=False)
    )
    outer_mesh = trimesh.Trimesh(
        *read_geometry(str(outer_surface), read_metadata=False)
    )

    edges = np.asarray(trimesh.graph.vertex_adjacency_graph(outer_mesh).edges)
    edges = np.concatenate([edges[:, [1, 0]], edges])
    robust_normals = [
        np.median(
            outer_mesh.vertex_normals[edges[edges[:, 0] == vertex_id][:, 1]],
            axis=0,
        )
        for vertex_id in np.arange(outer_mesh.vertices.shape[0])
    ]

    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(inner_mesh)
    intersections, indices = ray_intersector.intersects_location(
        ray_origins=outer_mesh.vertices,
        ray_directions=robust_normals,
        multiple_hits=True,
    )[:2]
    delta_1 = np.zeros(outer_mesh.vertices.shape[0])
    if len(indices):
        distance = np.sqrt(
            ((outer_mesh.vertices[indices] - intersections) ** 2).sum(axis=1)
        )
        keep = distance < MAX_CORRECTION_DISTANCE_MM
        indices, distance = indices[keep], distance[keep]
        if len(indices):
            delta_1[indices] = distance + move_margin_mm

    closest, distance, triangle_id = trimesh.proximity.closest_point(
        outer_mesh, inner_mesh.vertices
    )
    normal = (inner_mesh.vertices - closest) / distance[:, None]
    angles = trimesh.geometry.vector_angle(
        np.stack([normal, outer_mesh.face_normals[triangle_id]], axis=1)
    )

    keep = (angles < 1.0) & (distance < MAX_CORRECTION_DISTANCE_MM)
    proposed = (
        pd.DataFrame(
            {
                "vertex_id": outer_mesh.faces[triangle_id[keep]].ravel(),
                "delta": np.tile(distance[keep] + move_margin_mm, (3, 1)).T.ravel(),
            }
        )
        .groupby("vertex_id")
        .max()
        .reset_index()
    )
    delta_2 = np.zeros(outer_mesh.vertices.shape[0])
    if len(proposed):
        delta_2[proposed.vertex_id.to_numpy(dtype=int)] = (
            proposed.delta.to_numpy(dtype=float)
        )
    return np.stack([delta_1, delta_2]).max(axis=0)


def fix_intersecting_surfaces(
    inner_surface: Path,
    outer_surface: Path,
    move_margin_mm: float = MOVE_MARGIN_MM,
) -> np.ndarray | None:
    deltas = intersection_deltas(inner_surface, outer_surface, move_margin_mm)
    if np.all(deltas == 0):
        return None

    outer_vertices, outer_faces, volume_info = read_geometry(
        str(outer_surface), read_metadata=True
    )
    outer_mesh = trimesh.Trimesh(outer_vertices, outer_faces)
    moved_vertices = outer_mesh.vertices + (
        deltas[:, None] * outer_mesh.vertex_normals
    )
    write_geometry(
        str(outer_surface),
        moved_vertices,
        outer_mesh.faces,
        volume_info=volume_info,
    )
    return deltas


def correct_pial_intersections(subject_dir: Path, bem_dir: Path) -> dict[str, int]:
    inner_skull = bem_dir / "inner_skull.surf"
    movements = {}
    for hemisphere in ("lh", "rh"):
        deltas = fix_intersecting_surfaces(
            subject_dir / "surf" / f"{hemisphere}.pial", inner_skull
        )
        movements[f"{hemisphere}.pial_vs_inner_skull"] = (
            0 if deltas is None else int(np.count_nonzero(deltas))
        )
    return movements

# If MNE raises an error, below function identifies which surface pair caused it
def classify_mne_nesting_failure(error_text: str) -> str | None:
    for pair_name, pattern in NESTING_PATTERNS.items():
        if pattern.search(error_text):
            return pair_name
    return None


def attempt_mne_model_and_solution(
    subject: str, subjects_dir: Path, output_name: str
) -> dict[str, object]:
    result: dict[str, object] = {
        "model_status": "NOT_RUN",
        "solution_status": "NOT_RUN",
        "failure_stage": "",
        "error": "",
        "diagnosed_bem_pair": None,
    }
    try:
        model = mne.make_bem_model(
            subject=subject,
            subjects_dir=str(subjects_dir),
            conductivity=CONDUCTIVITY,
            ico=None,
            verbose=True,
        )
        result["model_status"] = "PASS"
        model_path = subjects_dir / subject / "bem" / f"{output_name}-bem.fif"
        mne.write_bem_surfaces(model_path, model, overwrite=True)
    except Exception:
        result["model_status"] = "FAIL"
        result["failure_stage"] = "make_bem_model"
        result["error"] = traceback.format_exc()
        result["diagnosed_bem_pair"] = classify_mne_nesting_failure(
            str(result["error"])
        )
        return result

    try:
        solution = mne.make_bem_solution(model, verbose=True)
        result["solution_status"] = "PASS"
        solution_path = (
            subjects_dir / subject / "bem" / f"{output_name}-sol.fif"
        )
        mne.write_bem_solution(solution_path, solution, overwrite=True)
    except Exception:
        result["solution_status"] = "FAIL"
        result["failure_stage"] = "make_bem_solution"
        result["error"] = traceback.format_exc()
    return result


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def create_subject(age: str) -> dict[str, object]:
    paths = input_paths(age)
    source_subject = str(paths["source_subject"])
    output_subject = output_subject_name(age)
    final_dir = OUTPUT_SUBJECTS_DIR / output_subject
    if final_dir.exists():
        if not OVERWRITE:
            raise FileExistsError(final_dir)
        backup = final_dir.with_name(
            f"{final_dir.name}.backup.{datetime.now():%Y%m%dT%H%M%S}"
        )
        final_dir.rename(backup)

    staging_subject = f".{output_subject}.staging.{os.getpid()}"
    staging_dir = OUTPUT_SUBJECTS_DIR / staging_subject
    if staging_dir.exists():
        raise FileExistsError(staging_dir)
    bem_dir = staging_dir / "bem"
    bem_dir.mkdir(parents=True)
    os.symlink(paths["subject_dir"] / "mri", staging_dir / "mri", target_is_directory=True)
    os.symlink(paths["subject_dir"] / "surf", staging_dir / "surf", target_is_directory=True)

    try:
        process_bem_volume(paths["bem_volume"], paths["lh_white"], bem_dir)
        """
        The BEM inner-skull boundary should enclose the cortical surfaces.
        Intersections between the pial surface and inner skull can indicate anatomically inconsistent geometry.
        So, we check the left and right pial surfaces against the inner-skull BEM surface.
        Where a correction is identified, the inner-skull mesh vertices are displaced outward along their normals.
        """
        pial_movements = correct_pial_intersections(staging_dir, bem_dir)
        mne_result = attempt_mne_model_and_solution(
            staging_subject, OUTPUT_SUBJECTS_DIR, output_subject
        )

        if mne_result["model_status"] == "PASS" and mne_result["solution_status"] == "PASS":
            status = "PASS"
        elif mne_result["diagnosed_bem_pair"] is not None:
            status = "NEEDS_BEM_NESTING_CORRECTION"
        elif mne_result["failure_stage"] == "make_bem_solution":
            status = "SOLUTION_FAILURE_NOT_AN_INTERSECTION_DIAGNOSIS"
        else:
            status = "MNE_FAILURE_NOT_CLASSIFIED_AS_BEM_NESTING"

        record = {
            "schema_version": 2,
            "age": age,
            "source_subject": source_subject,
            "output_subject": output_subject,
            "surface_logic": "O'Reilly BEM extraction with declared project differences",
            "target_triangles": TARGET_TRIANGLES,
            "conductivity_S_per_m": list(CONDUCTIVITY),
            "pial_correction": pial_movements,
            "mne": mne_result,
            "status": status,
        }
        write_json(bem_dir / "creation_status.json", record)
        staging_dir.rename(final_dir)
        return record
    except Exception:
        failed_dir = OUTPUT_SUBJECTS_DIR / (
            f"{output_subject}.FAILED.{datetime.now():%Y%m%dT%H%M%S}"
        )
        if staging_dir.exists():
            staging_dir.rename(failed_dir)
        raise


def write_summary(rows: list[dict[str, object]]) -> None:
    path = OUTPUT_SUBJECTS_DIR / "bem_creation_summary.tsv"
    fields = (
        "age",
        "source_subject",
        "output_subject",
        "status",
        "model_status",
        "solution_status",
        "diagnosed_bem_pair",
        "error",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in rows:
            mne_result = record.get("mne", {})
            writer.writerow(
                {
                    "age": record.get("age", ""),
                    "source_subject": record.get("source_subject", ""),
                    "output_subject": record.get("output_subject", ""),
                    "status": record.get("status", ""),
                    "model_status": mne_result.get("model_status", ""),
                    "solution_status": mne_result.get("solution_status", ""),
                    "diagnosed_bem_pair": mne_result.get("diagnosed_bem_pair", ""),
                    "error": str(mne_result.get("error", "")).replace("\n", " | "),
                }
            )


def main() -> int:
    preflight()
    OUTPUT_SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, age in enumerate(AGES, start=1):
        print(f"[{index}/{len(AGES)}] ANTS{age}Years3T", flush=True)
        try:
            record = create_subject(age)
        except Exception:
            record = {
                "age": age,
                "source_subject": f"ANTS{age}Years3T",
                "output_subject": output_subject_name(age),
                "status": "EXTRACTION_FAILED",
                "mne": {"error": traceback.format_exc()},
            }
        rows.append(record)
        write_summary(rows)

    failed = [
        row
        for row in rows
        if row["status"] not in {"PASS", "NEEDS_BEM_NESTING_CORRECTION"}
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
References:
Richards JE, Sanchez C, Phillips-Meek M, Xie W. A database of age-appropriate average MRI templates. Neuroimage. 2016;124(Pt B):1254-1259. doi:10.1016/j.neuroimage.2015.04.055
Christian O'Reilly, Eric Larson, John E. Richards, Mayada Elsabbagh, Structural templates for imaging EEG cortical sources in infants, NeuroImage, Volume 227, 2021, 117682, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2020.117682.
Dogdas B, Shattuck DW, Leahy RM. Segmentation of skull and scalp in 3-D human MRI using mathematical morphology. Hum Brain Mapp. 2005;26(4):273-285. doi:10.1002/hbm.20159
"""
