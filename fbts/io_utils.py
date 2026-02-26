from __future__ import annotations

from pathlib import Path
from typing import List

from .model import Site


def append_xyz_frame(path: Path, sites: List[Site], comment: str) -> None:
    visible_sites = [site for site in sites if site.site_type != "H"]
    lines = [str(len(visible_sites)), comment]
    for site in visible_sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_initial_xyz(path: Path, sites: List[Site], comment: str) -> None:
    visible_sites = [site for site in sites if site.site_type != "H"]
    lines = [str(len(visible_sites)), comment]
    for site in visible_sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
