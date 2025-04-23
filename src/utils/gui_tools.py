import logging
from .file_utils import read_file, write_file
from fontTools.colorLib import geometry

logger = logging.getLogger(__name__)

def load_geometry(root, geometry_file):
    logger.debug(f"Loading geometry from {geometry_file}")
    try:
        geometry = read_file(geometry_file)
        root.geometry(geometry)
    except Exception:
        geometry = "1900x1200"
    return geometry

def save_geometry(root, geometry_file):
    logger.debug(f"Saving geometry to {geometry_file}")
    geometry = root.geometry()
    write_file(geometry_file, geometry)

def center_popup(parent, child):
    parent.update_idletasks()
    x = parent.winfo_x() + parent.winfo_width() // 2 - child.winfo_reqwidth() // 2
    y = parent.winfo_y() + parent.winfo_height() // 2 - child.winfo_reqheight() // 2
    child.geometry(f"+{x}+{y}")
