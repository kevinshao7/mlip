"""Local trajectory parsing and molecular-component helpers for the X campaign."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic

LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


@dataclass(frozen=True)
class Frame:
    symbols: tuple[str, ...]
    positions: np.ndarray
    cell: np.ndarray
    comment: str


def iter_xyz_frames(path: Path):
    """Yield frames from an extxyz trajectory without loading it into memory."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            atom_count_line = handle.readline()
            while atom_count_line and not atom_count_line.strip():
                atom_count_line = handle.readline()
            if not atom_count_line:
                return
            try:
                natoms = int(atom_count_line.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid atom count at frame {frame_index} in {path}") from exc
            comment = handle.readline().strip()
            lattice_match = LATTICE_RE.search(comment)
            if lattice_match is None:
                raise ValueError(f"Frame {frame_index} in {path} has no Lattice field")
            lattice = np.fromstring(lattice_match.group(1), sep=" ")
            if lattice.size != 9:
                raise ValueError(f"Frame {frame_index} in {path} has an invalid Lattice field")
            symbols = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                line = handle.readline()
                if not line:
                    raise ValueError(f"Incomplete frame {frame_index} in {path}")
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed atom {atom_index} in frame {frame_index} of {path}")
                symbols.append(fields[0])
                positions[atom_index] = [float(value) for value in fields[1:4]]
            yield frame_index, Frame(tuple(symbols), positions, lattice.reshape(3, 3), comment)
            frame_index += 1


def minimum_image_distances(first_positions: np.ndarray, second_positions: np.ndarray,
                            cell: np.ndarray) -> np.ndarray:
    delta = first_positions[:, None, :] - second_positions[None, :, :]
    shape = delta.shape[:2]
    _vectors, distances = find_mic(delta.reshape(-1, 3), cell=cell, pbc=True)
    return distances.reshape(shape)


def pair_cutoff(first: str, second: str, oh_cutoff: float, nh_cutoff: float,
                hh_cutoff: float, bond_scale: float) -> float:
    pair = frozenset((first, second))
    if pair == {"H", "O"}:
        return oh_cutoff
    if pair == {"H", "N"}:
        return nh_cutoff
    if first == second == "H":
        return hh_cutoff
    return bond_scale * (covalent_radii[atomic_numbers[first]] + covalent_radii[atomic_numbers[second]])


def molecular_components(frame: Frame, oh_cutoff: float, nh_cutoff: float,
                         hh_cutoff: float, bond_scale: float) -> tuple[list[list[int]], float]:
    """Assign every H to its nearest H/N/O atom and identify heavy-atom bonds."""
    natoms = len(frame.symbols)
    adjacency = [set() for _ in range(natoms)]
    minimum_distance = np.inf
    for left in range(natoms - 1):
        distances = minimum_image_distances(
            frame.positions[left:left + 1], frame.positions[left + 1:], frame.cell
        )[0]
        minimum_distance = min(minimum_distance, float(distances.min()))
        for offset, distance in enumerate(distances, start=left + 1):
            if "H" in (frame.symbols[left], frame.symbols[offset]):
                continue
            if distance <= pair_cutoff(frame.symbols[left], frame.symbols[offset],
                                       oh_cutoff, nh_cutoff, hh_cutoff, bond_scale):
                adjacency[left].add(offset)
                adjacency[offset].add(left)
    eligible = np.array(
        [index for index, symbol in enumerate(frame.symbols) if symbol in {"H", "O", "N"}],
        dtype=int,
    )
    for hydrogen, symbol in enumerate(frame.symbols):
        if symbol != "H":
            continue
        candidates = eligible[eligible != hydrogen]
        if candidates.size:
            distances = minimum_image_distances(
                frame.positions[hydrogen:hydrogen + 1], frame.positions[candidates], frame.cell
            )[0]
            nearest = int(candidates[int(np.argmin(distances))])
            adjacency[hydrogen].add(nearest)
            adjacency[nearest].add(hydrogen)
    unseen = set(range(natoms))
    components = []
    while unseen:
        start = unseen.pop()
        component = [start]
        stack = [start]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.append(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components, float(minimum_distance)


def component_formula(frame: Frame, component: list[int]) -> str:
    counts = Counter(frame.symbols[index] for index in component)
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(counts.keys() - {"C", "H"})
    else:
        order = sorted(counts)
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else "") for symbol in order)
