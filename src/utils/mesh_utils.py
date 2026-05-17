import json
import os
import shutil
from pathlib import Path

import meshio
import numpy as np


def convert_to_xdmf(mesh_dir, output_file):
    """Converts a custom mesh format to xdmf format

    Args:
        mesh_dir: Directory path containing mesh.json and nodeX.npy files
        output_file: Path to the output xdmf file
    """
    # Read node coordinates
    nodes = np.load(os.path.join(mesh_dir, "nodeX.npy"))
    print(f"Node coordinate shape: {nodes.shape}")

    # Ensure node coordinates are 2D or 3D
    if len(nodes.shape) == 1:
        nodes = nodes.reshape(-1, 2)  # Assume it's a 2D mesh
    elif len(nodes.shape) > 2:
        raise ValueError(f"Incorrect node coordinate dimensions: {nodes.shape}")

    # Read element connectivity
    with open(os.path.join(mesh_dir, "mesh.json")) as f:
        elements = json.load(f)
    print(f"Element connectivity type: {type(elements)}")
    print(f"Element connectivity length: {len(elements)}")

    # Ensure elements is a list or array
    if isinstance(elements, list | np.ndarray):
        elements = np.array(elements)
        if len(elements.shape) == 1:
            # If it's a 1D array, reshape it to a 2D array
            print(f"Elements shape before reshaping: {elements.shape}")
            elements = elements.reshape(-1, 3)  # Assume each element has 3 nodes
            print(f"Elements shape after reshaping: {elements.shape}")
    else:
        raise ValueError(f"Incorrect element connectivity format: {type(elements)}")

    # Ensure node numbering starts from 0
    if elements.min() > 0:
        elements = elements - 1

    # Print mesh information
    print("\n=== Mesh Information ===")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of elements: {len(elements)}")
    print(f"Element shape: {elements.shape}")

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=nodes,
        cells=[("triangle", elements)],  # Assume triangle cells
    )

    # Save as xdmf file
    # Keep only x and y coordinates
    mesh.points = mesh.points[:, :2]
    mesh.write(output_file)
    print(f"\nMesh saved to: {os.path.abspath(output_file)}")

    return nodes, elements


def get_dataset_mesh_path(data_dir: str, geometry: str, material: int, force_type: str = "shear") -> Path:
    """
    Build original mesh file path based on dataset parameters

    Args:
        data_dir: Dataset root directory
        geometry: Geometry type (e.g., 'C')
        material: Material ID (e.g., 0)
        force_type: Force type (e.g., 'shear')

    Returns:
        Directory path containing mesh.json and nodeX.npy
    """
    data_dir_path = Path(data_dir)

    # Build path based on dataset type
    if "train" in str(data_dir_path):
        # Training dataset path format
        mesh_path = data_dir_path / f"Model-{geometry}" / f"Model-{geometry}-{force_type}-Poly-{material:05d}"
    elif "valid-10" in str(data_dir_path):
        # valid-10 dataset path format
        mesh_path = data_dir_path / f"{force_type}-{material:05d}"
    else:
        # Other validation dataset path format
        mesh_path = data_dir_path / f"Model-{geometry}" / f"Model-{geometry}-{force_type}-{material:05d}"

    print(f"Searching for mesh file path: {mesh_path}")
    return mesh_path


def load_mesh_from_dataset(
    data_dir: str, geometry: str, material: int, force_type: str = "shear", output_dir: Path = None
) -> tuple[Path, np.ndarray, np.ndarray]:
    """
    Load mesh from dataset and return XDMF file path, nodes and elements data

    Args:
        data_dir: Dataset root directory
        geometry: Geometry type
        material: Material ID
        force_type: Force type
        output_dir: Output directory, if None use temporary directory

    Returns:
        (xdmf_file_path, nodes_array, elements_array)
    """
    # Get original mesh file path
    mesh_path = get_dataset_mesh_path(data_dir, geometry, material, force_type)

    # Read mesh data directly from source
    mesh_json = mesh_path / "mesh.json"
    node_x_file = mesh_path / "nodeX.npy"

    # Read mesh.json
    with open(mesh_json) as f:
        mesh_data = json.load(f)

    # Read nodeX.npy
    nodes = np.load(node_x_file)

    # Extract elements data
    elements = np.array(mesh_data)

    # Generate XDMF file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy mesh files to output directory
    required_files = ["mesh.json", "nodeX.npy", "nodeU.npy"]
    optional_files = ["elementS.npy", "lam3.npy"]

    for file_name in required_files:
        src_file = mesh_path / file_name
        dst_file = output_dir / file_name
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")

    for file_name in optional_files:
        src_file = mesh_path / file_name
        if src_file.exists():
            dst_file = output_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"Copied (optional): {src_file} -> {dst_file}")
        else:
            print(f"(Optional) file not found, skip: {src_file}")

    # Generate XDMF file
    xdmf_file = output_dir / "mesh.xdmf"
    print(f"Generating XDMF file: {xdmf_file}")
    convert_to_xdmf(str(output_dir), str(xdmf_file))

    print("Successfully loaded mesh data:")
    print(f"  XDMF file: {xdmf_file}")
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of elements: {len(elements)}")

    return xdmf_file, nodes, elements


def cleanup_temp_mesh_files(xdmf_file: Path):
    """
    Clean up temporary mesh files

    Args:
        xdmf_file: XDMF file path
    """
    if xdmf_file.exists():
        mesh_dir = xdmf_file.parent
        # Delete all files in temporary directory
        for file_path in mesh_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                print(f"Deleted temporary file: {file_path}")

        # Delete temporary directory
        mesh_dir.rmdir()
        print(f"Deleted temporary directory: {mesh_dir}")
