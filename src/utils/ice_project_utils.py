"""
this file contains utils that are project-specific
"""

import ast


def get_mesh_id(description: str) -> tuple[str, list[dict]]:
    """
    get the material id and meshlist ids from the description
    example usage:
    material_id, meshlist_ids = get_mesh_id(meshlist.description)
    for i, mesh in enumerate(meshlist):
        f"Mesh {material_id}_{meshlist_ids[i]['geometry']}_{meshlist_ids[i]['snapshot']}"
    """
    material_id = description.split("material id: ")[1].split(",")[0]
    meshlist_id_str = description.split("meshlist_id: ")[1]
    meshlist_ids = ast.literal_eval(meshlist_id_str)
    return material_id, meshlist_ids
