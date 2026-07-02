# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Potential energy function loading and construction."""

from typing import Any, Callable, Optional, Tuple, List

import chex
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

from jax_md import energy, partition, space
from jax_md.util import Array

from .config import (
    ClassicalPotentialConfig,
    NeighborListConfig,
    NNPotentialConfig,
)


def replicate_params_to_mesh(params: Any, mesh: Mesh) -> Any:
    """Replicate model parameters across all devices in a mesh.
    
    This is required for shard_map parallelism where each device needs
    access to the full model parameters (they're captured in closures).
    
    Args:
        params: Model parameters pytree.
        mesh: The JAX mesh to replicate across. Must be the SAME mesh
              used for shard_map operations.
        
    Returns:
        Replicated params accessible from all devices in the mesh.
    """
    # Create a sharding that replicates across all devices (no partitioning = P())
    replicated_sharding = NamedSharding(mesh, P())
    # Put params on all devices
    return jax.tree.map(
        lambda x: jax.device_put(x, replicated_sharding),
        params
    )


def bind_lambda_energy_fn(
    base_energy_fn: Callable,
    lambda_val: float,
    decouple_indices: Optional[Array] = None,
    lambda_electrostatics: Optional[float] = None,
    total_charge_a: Optional[float] = None,
    total_spin_a: Optional[float] = None,
    total_charge_b: Optional[float] = None,
    total_spin_b: Optional[float] = None,
) -> Callable:
    """Create an energy function bound to a specific lambda value.

    This is useful for replica exchange where each replica runs at a fixed lambda
    and needs an energy function compatible with jax-md simulators (which don't
    know about lambda_val).

    The lambda value is passed through to the underlying energy function (e.g., MACE)
    which handles alchemical transformations internally.

    Args:
        base_energy_fn: The base energy function that accepts lambda_val as a kwarg.
        lambda_val: The lambda value to bind (in [0, 1]).
        decouple_indices: Optional array of atom indices being alchemically decoupled.
            Used with lambda_val for alchemical free energy calculations.
        lambda_electrostatics: Optional lambda for electrostatic scaling [0,1].
            If None, the model will default to using lambda_val for electrostatics.
            Only used by PolarMACE models.
        total_charge_a: Optional charge of fragment A (non-decoupled) at lambda=0.
        total_spin_a: Optional spin of fragment A at lambda=0.
        total_charge_b: Optional charge of fragment B (decoupled) at lambda=0.
        total_spin_b: Optional spin of fragment B at lambda=0.

    Returns:
        An energy function with signature energy_fn(positions, *, neighbor, **kwargs)
        that passes the bound lambda value to the base function.
    """
    def energy_fn(positions, *, neighbor, **kwargs):
        """Energy function with bound lambda value passed to underlying potential."""
        return base_energy_fn(
            positions,
            neighbor=neighbor,
            lambda_val=lambda_val,
            decouple_indices=decouple_indices,
            lambda_electrostatics=lambda_electrostatics,
            total_charge_a=total_charge_a,
            total_spin_a=total_spin_a,
            total_charge_b=total_charge_b,
            total_spin_b=total_spin_b,
            **kwargs
        )
    return energy_fn


def box_vec_inverse(mat):
    """Fast inversion of `mat=box_vectors` given that we know it is lower triangular (ASE convention).
    
    Always computed in float64 for numerical stability.
    """
    chex.assert_shape(mat, (3, 3))
    mat_f64 = mat.astype(jnp.float64)
    return jax.scipy.linalg.solve_triangular(mat_f64, jnp.eye(3, dtype=jnp.float64), lower=True)


def invert_pos(pos, box):
    """Map Cartesian positions into fractional coordinates [0,1].
    
    Always computed in float64 for numerical stability in periodic image determination.
    
    Args:
        pos: Cartesian positions [N, 3]
        box: jax-md box matrix (column vectors = transposed ASE cell, upper triangular)
    
    Returns:
        Fractional coordinates [N, 3] in float64
    """
    # box is in jax-md convention (column vectors). 
    # To get fractional coords: pos_frac = pos @ inv(cell) where cell = box.T (ASE convention)
    # So: pos_frac = pos @ inv(box.T) = pos @ (box^{-1}).T = pos @ inv_box_ase
    pos_f64 = pos.astype(jnp.float64)
    box_f64 = box.astype(jnp.float64)
    box_ase = box_f64.T  # Convert to ASE convention (lower triangular)
    inv_box_ase = box_vec_inverse(mat=box_ase)  # Invert the ASE-convention matrix (already f64)
    return jnp.matmul(pos_f64, inv_box_ase, precision="highest")


def compute_minimum_image_vectors(
    positions: Array,
    senders: Array,
    receivers: Array,
    box: Array,
    edge_mask: Array,
) -> Array:
    """Compute minimum image displacement vectors for edge pairs.
    
    Geometry is computed in float64 for numerical stability when determining
    which periodic image to use. Results are returned in float32 for MACE.
    
    Args:
        positions: Cartesian positions [N, 3], any dtype
        senders: Sender atom indices [E], int
        receivers: Receiver atom indices [E], int  
        box: jax-md box matrix (column vectors), any dtype
        edge_mask: Boolean mask for valid edges [E]
        
    Returns:
        vectors: Minimum image displacement vectors [E, 3] in float32
            vectors[e] = positions[receivers[e]] - positions[senders[e]] + periodic_correction
    """
    n = positions.shape[0]
    
    # Clip indices for safe indexing
    senders_safe = jnp.clip(senders, 0, n - 1)
    receivers_safe = jnp.clip(receivers, 0, n - 1)
    
    # =========================================================================
    # High-precision geometry (float64) for periodic image determination
    # This is critical - small errors in determining which periodic image
    # to use can cause large energy errors and simulation instability.
    # =========================================================================
    positions_f64 = positions.astype(jnp.float64)
    box_f64 = box.astype(jnp.float64)
    
    # Convert positions to fractional coordinates
    pos_frac = invert_pos(positions_f64, box_f64)
    
    # Fractional difference -> round to get integer periodic shifts
    dR_frac = pos_frac[receivers_safe] - pos_frac[senders_safe]
    shift_int = jnp.round(dR_frac)  # Integer shifts in fractional coords
    
    # Direct Cartesian difference
    direct_vecs = positions_f64[receivers_safe] - positions_f64[senders_safe]
    
    # Transform integer shifts to real space correction
    # box has lattice vectors as columns, so shift @ box.T gives real-space shift
    shift_real = jnp.matmul(shift_int, box_f64.T, precision="highest")
    
    # Minimum image vector = direct - periodic_correction
    vectors_f64 = direct_vecs - shift_real
    
    # Mask invalid edges
    vectors_f64 = jnp.where(edge_mask[:, None], vectors_f64, 0.0)
    
    # Convert to float32 for MACE (which was trained in float32)
    return vectors_f64.astype(jnp.float32)

def sort_edges_by_receiver(
    edges: tuple[chex.Array, chex.Array], N: int, pre_sorted: str | None = None, vectors: chex.Array | None = None
) -> tuple[tuple[chex.Array, chex.Array], chex.Array, chex.Array | None]:
    """
    Sorts edges by receiver, then by sender (secondary).

    Args:
        edges: Tuple of (senders, receivers), each int32 array of shape [E].
        N: number of nodes (used for constructing sort key).
        pre_sorted: Optional string indicating if edges are already sorted ("receiver" or "sender").
        vectors: Optional float32 array of shape [E, 3] to be reordered along with edges.

    Returns a tuple containing:
      (senders_sorted, receivers_sorted): Tuple of int32 arrays, each [E]
      segments: [N+1] int32   -- CSR segment offsets (starts at 0, ends at E)
      vectors_sorted: [E, 3] float32 | None   -- [vectors_sorted] or None if vectors are not provided
    """
    assert isinstance(edges, tuple) and len(edges) == 2
    senders, receivers = edges
    senders = senders.astype(jnp.int32)
    receivers = receivers.astype(jnp.int32)

    n_edges = senders.shape[0]
    chex.assert_equal_shape((senders, receivers))
    chex.assert_rank(senders, 1)
    if vectors is not None:
        chex.assert_shape(vectors, (n_edges, 3))

    if n_edges == 0:
        return (
            (jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)),
            compute_csr_from_sorted_indices(jnp.array([], dtype=jnp.int32), N),
            None if vectors is None else jnp.array([], dtype=float).reshape(0, 3),
        )

    if pre_sorted == "receiver":
        senders_sorted = senders
        receivers_sorted = receivers
    elif pre_sorted == "sender":
        senders_sorted = receivers
        receivers_sorted = senders
        vectors = -vectors if vectors is not None else None
    else:
        # Primary by receiver, secondary by sender
        key = receivers * N + senders
        order = jnp.argsort(key)
        senders_sorted = senders[order]
        receivers_sorted = receivers[order]
        vectors = vectors[order] if vectors is not None else None

    segments = compute_csr_from_sorted_indices(
        receivers_sorted, N
    )  # we pass receivers because we are sorting by receiver

    return (senders_sorted, receivers_sorted), segments, vectors

def compute_csr_from_sorted_indices(indices_sorted: chex.Array, N: int) -> chex.Array:
    """
    Computes CSR segment offsets from sorted edges.
     If edges are sorted by sender, the input: ``indices_sorted`` should be senders.
     The opposite is true for receiver sorted edges.

    Args:
        indices_sorted: [E] int32
        N: number of nodes.

    Returns:
      segments: [N+1] int32   -- CSR segment offsets (starts at 0, ends at E)
    """
    if len(indices_sorted) == 0:
        return jnp.array([0] * (N + 1), dtype=jnp.int32)

    # CSR offsets: counts per receiver, then exclusive cumsum
    counts = jnp.bincount(indices_sorted, length=N)
    return jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)], axis=0)


def get_neighbor_list_format(format_str: str) -> partition.NeighborListFormat:
    """Convert format string to partition.NeighborListFormat."""
    formats = {
        'Dense': partition.Dense,
        'Sparse': partition.Sparse,
        'OrderedSparse': partition.OrderedSparse,
    }
    if format_str not in formats:
        raise ValueError(
            f'Unknown neighbor list format: {format_str}. '
            f'Must be one of {list(formats.keys())}'
        )
    return formats[format_str]


def _load_mace_off_checkpoint(checkpoint_path: str, species: Array,
                              mesh_for_replication=None):
    """Load a MACE-OFF model from checkpoint.

    Returns:
        model: MACE flax module instance.
        params: Model parameters.
        species_indices: Species index array for the given atomic numbers.
        r_cutoff: Model r_max cutoff.
        mace_cfg: Full model config dict.
    """
    from aai_mace.mace_jax.models.mace_off import MACE
    from pathlib import Path
    import json
    import orbax.checkpoint as ocp

    save_dir = Path(checkpoint_path).expanduser().resolve()

    # Load model config FIRST to get r_max and avg_num_neighbors
    model_config_path = save_dir / "model_config.json"
    if model_config_path.exists():
        # Converted format: model_config.json in checkpoint root
        with open(model_config_path) as f:
            mace_cfg = json.load(f)
        mace_cfg.pop('atomic_numbers', None)
        atom_energies_dict = None
        atomic_numbers_in_model = None
        if mace_cfg.get('atom_energies') and isinstance(mace_cfg['atom_energies'], dict):
            atom_energies_dict = {int(k): float(v) for k, v in mace_cfg['atom_energies'].items()}
            atomic_numbers_in_model = sorted(atom_energies_dict.keys())
            atom_energies_list = [atom_energies_dict[z] for z in atomic_numbers_in_model]
            num_species = mace_cfg.get('num_species', len(atom_energies_list))
            while len(atom_energies_list) < num_species:
                atom_energies_list.append(0.0)
            mace_cfg['atom_energies'] = atom_energies_list
        ckpt_subdirs = [d for d in save_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if ckpt_subdirs:
            ckpt_path = max(ckpt_subdirs, key=lambda d: int(d.name))
        else:
            ckpt_path = save_dir
    else:
        from aai_mace.experiments.load_checkpoint import _load_mace_config, _resolve_checkpoint_path
        ckpt_path = _resolve_checkpoint_path(save_dir)
        cfg_root = save_dir if (save_dir / "experiment_info.json").exists() else save_dir.parent.parent
        mace_cfg = _load_mace_config(cfg_root, None)
        atom_energies_dict = None
        atomic_numbers_in_model = None
        if mace_cfg.get('atom_energies') and isinstance(mace_cfg['atom_energies'], dict):
            atom_energies_dict = {int(k): float(v) for k, v in mace_cfg['atom_energies'].items()}
            atomic_numbers_in_model = sorted(atom_energies_dict.keys())
            atom_energies_list = [atom_energies_dict[z] for z in atomic_numbers_in_model]
            num_species = mace_cfg.get('num_species', len(atom_energies_list))
            while len(atom_energies_list) < num_species:
                atom_energies_list.append(0.0)
            mace_cfg['atom_energies'] = atom_energies_list

    if 'r_max' not in mace_cfg:
        raise ValueError(
            f"Model config missing 'r_max'. Cannot determine neighbor list cutoff. "
            f"Checkpoint: {save_dir}"
        )
    r_cutoff = mace_cfg['r_max']

    if 'avg_num_neighbors' not in mace_cfg:
        raise ValueError(
            f"Model config missing 'avg_num_neighbors'. This is required for correct "
            f"energy normalization. Checkpoint may be corrupted: {save_dir}"
        )
    if mace_cfg['avg_num_neighbors'] == 1.0:
        raise ValueError(
            f"Model has avg_num_neighbors=1.0 which is likely incorrect. "
            f"This usually indicates the model was not properly trained or the "
            f"checkpoint is incomplete. Checkpoint: {save_dir}"
        )

    valid_mace_keys = {
        'avg_num_neighbors', 'r_max', 'hidden_width', 'message_l',
        'readout_mlp_width', 'correlation', 'num_bessel', 'num_polynomial_cutoff',
        'mlp_hidden', 'scale', 'shift', 'num_species', 'num_interactions',
        'atom_energies', 'lmbda_update_attrs', 'lmbda_update_feats',
        'linear_precision', 'tp_precision', 'mlp_precision', 'contraction_precision',
    }
    filtered_cfg = {k: v for k, v in mace_cfg.items() if k in valid_mace_keys}
    removed_keys = set(mace_cfg.keys()) - valid_mace_keys
    if 'density_interaction_blocks' in removed_keys:
        if mace_cfg['density_interaction_blocks']:
            raise ValueError(
                f"Checkpoint uses density_interaction_blocks=True which is not supported. "
                f"Checkpoint: {save_dir}"
            )
        removed_keys.discard('density_interaction_blocks')
    if removed_keys:
        print(f'  Ignoring unsupported config keys: {removed_keys}')

    model = MACE(**filtered_cfg)

    if model_config_path.exists():
        dummy_params = model.init(
            jax.random.PRNGKey(0),
            positions=jnp.zeros((2, 3), dtype=jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32),
            node_species=jnp.array([0, 0]),
            edges_sorted_by_receiver=(jnp.array([1, 0], dtype=jnp.int32), jnp.array([0, 1], dtype=jnp.int32)),
            segments=jnp.array([0, 1, 2]),
        )
        options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_manager = ocp.CheckpointManager(save_dir, options=options)
        step = checkpoint_manager.latest_step()
        params = checkpoint_manager.restore(step, args=ocp.args.StandardRestore(dummy_params))
    else:
        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(str(ckpt_path))
        if isinstance(state, dict):
            params = state.get('ema_params', state.get('params', state))
        else:
            params = state

    if mesh_for_replication is not None:
        params = replicate_params_to_mesh(params, mesh_for_replication)

    if atom_energies_dict is not None:
        atomic_num_to_species_idx = {z: i for i, z in enumerate(atomic_numbers_in_model)}
        species_indices = jnp.array([atomic_num_to_species_idx[int(z)] for z in species])
    else:
        raise ValueError(
            f"model_config.json has 'atom_energies' as a list instead of a dict keyed "
            f"by atomic number (e.g. {{\"1\": -13.57, \"6\": -1030.56, ...}}). "
            f"Using raw atomic numbers as species indices would produce incorrect results. "
            f"Re-run the conversion script to fix the checkpoint format. "
            f"Checkpoint: {save_dir}"
        )

    return model, params, species_indices, r_cutoff, mace_cfg


def load_nn_potential(
    config: NNPotentialConfig,
    displacement_fn: Callable,
    box: Array | None,
    species: Array,
    neighbor_config: NeighborListConfig,
    positions: Array,
    fractional_coordinates: bool = True,
    mesh_for_replication: Optional[Mesh] = None,
) -> Tuple[Callable, Callable, Any]:
    """Load neural network potential from checkpoint.

    Args:
      config: Neural network potential configuration.
      displacement_fn: Displacement function from space module.
      box: Simulation box.
      species: Atomic numbers array.
      neighbor_config: Neighbor list configuration.
      positions: Initial atomic positions (used for neighbor list estimation).
      fractional_coordinates: Whether using fractional coordinates.
      mesh_for_replication: Optional JAX Mesh for replicating model parameters.
          Required for shard_map parallelism where each device needs access to the
          full model params. Must be the SAME mesh used for shard_map operations.
          If None, params stay on default device.

    Returns:
      neighbor_fn: Function to create/update neighbor lists.
      energy_fn: Energy function compatible with jax-md.
      params: Model parameters (for checkpointing).
    """
    model_type = config.model_type
    checkpoint_path = config.checkpoint_path
    nl_backend = neighbor_config.backend

    if model_type == 'mace_off':
        # Load base MACE-OFF model
        print(f'Loading base MACE-OFF model from {checkpoint_path}')
        model, params, species_indices, r_cutoff, mace_cfg = _load_mace_off_checkpoint(
            checkpoint_path, species, mesh_for_replication
        )
        print(f'Using model r_max = {r_cutoff} Å for neighbor list cutoff')
        print(f'Using model avg_num_neighbors = {mace_cfg["avg_num_neighbors"]:.2f}')

        # Load optional delta MACE-OFF model
        delta_model = None
        delta_params = None
        delta_species_indices = None
        nl_r_cutoff = r_cutoff  # Cutoff used for the shared neighbor list
        if config.delta_checkpoint_path:
            print(f'Loading delta MACE-OFF model from {config.delta_checkpoint_path}')
            delta_model, delta_params, delta_species_indices, delta_r_cutoff, delta_cfg = (
                _load_mace_off_checkpoint(
                    config.delta_checkpoint_path, species, mesh_for_replication
                )
            )
            print(f'Delta model r_max = {delta_r_cutoff} Å, '
                  f'avg_num_neighbors = {delta_cfg["avg_num_neighbors"]:.2f}')
            # Use the larger cutoff for the shared neighbor list
            nl_r_cutoff = max(r_cutoff, delta_r_cutoff)
            if nl_r_cutoff > r_cutoff:
                print(f'Using delta r_max = {nl_r_cutoff} Å for shared neighbor list '
                      f'(larger than base r_max = {r_cutoff} Å)')

        # =====================================================================
        # Neighbor list setup (nvalchemi only)
        # =====================================================================
        box_for_nl = box if box is not None else jnp.eye(3)

        from functools import partial
        from jax_md_cli.nvalchemi_nl import (
            estimate_inputs_to_neighbor_list,
            neighbor_list_jax_jit,
            estimate_cell_list_params,
        )
        print(f'Using nvalchemi neighbor list backend')

        # Resolve NL algorithm: 'auto' picks cell_list for large systems.
        algo = neighbor_config.algorithm
        if algo == 'auto':
            algo = (
                'cell_list'
                if positions.shape[0] > neighbor_config.algorithm_auto_threshold
                else 'naive'
            )
        print(f'  nvalchemi NL algorithm: {algo}')

        cell_for_nl = box_for_nl.T.astype(jnp.float32)
        positions_f32 = positions.astype(jnp.float32)

        max_neighbors, max_neighbors_per_atom, shifts = estimate_inputs_to_neighbor_list(
            positions_f32,
            nl_r_cutoff,
            cell=cell_for_nl,
            assume_mic=False,
        )
        # max_neighbors = 20000
        # max_neighbors_per_atom=500
        print(f'  max_neighbors={max_neighbors}, max_neighbors_per_atom={max_neighbors_per_atom}')

        max_total_cells = None
        if algo == 'cell_list':
            max_total_cells = estimate_cell_list_params(positions_f32, nl_r_cutoff, cell_for_nl)
            print(f'  max_total_cells={max_total_cells}')

        # Mutable state for the NL budget; updated by NvalchemiNeighborFn.allocate().
        nl_state = {
            'fn': partial(
                neighbor_list_jax_jit,
                shifts=shifts,
                cutoff=nl_r_cutoff,
                max_neighbors_per_atom=max_neighbors_per_atom,
                max_neighbors=max_neighbors,
                fill_value=0,
                algorithm=algo,
                max_total_cells=max_total_cells,
            ),
            'K': max_neighbors_per_atom,
            'M': max_neighbors,
            'shifts': shifts,
            'algo': algo,
            'max_total_cells': max_total_cells,
            'current_box': box_for_nl,
        }

        def _wrap_to_box_f32(pos_f32, box):
            """Wrap f32 Cartesian positions to [0, box] via integer cell shifts.

            box has COLUMNS as box vectors (JAX-MD convention: box = cell_ase.T).
            pos = frac @ box.T so frac = pos @ inv(box.T).  Gradient w.r.t.
            pos_f32 is the identity (floor is piecewise-constant).  Pass
            stop_gradient(box) if box gradients must not flow through wrapping.
            """
            box_f32 = jnp.asarray(box, dtype=jnp.float32)
            # pos = frac @ box.T → frac = pos @ inv(box.T); pos_wrapped = (frac%1) @ box.T
            inv_boxT = jnp.linalg.inv(box_f32.T)
            frac = pos_f32 @ inv_boxT
            return pos_f32 - jnp.floor(frac) @ box_f32.T

        def _nvalchemi_update(new_positions, box=None):
            """Run NL and return NvalchemiNbrs with preventive overflow flag.

            Positions are wrapped to [0, box] before passing to nvalchemi so
            that the returned unit_shifts are consistent with _wrap_to_box_f32
            used in energy_fn.  Sets did_buffer_overflow when any atom uses
            >= 90% of its per-atom budget (issue #13 A).
            """
            _box = box if box is not None else nl_state['current_box']
            if _box is None:
                raise ValueError(
                    "MACE (nvalchemi) potentials require a periodic box. "
                    "Non-periodic systems (box=None) are not supported."
                )
            nl_state['current_box'] = _box
            _cell = jax.lax.stop_gradient(_box.T.astype(jnp.float32))
            _pos = jax.lax.stop_gradient(new_positions.astype(jnp.float32))
            _pos = _wrap_to_box_f32(_pos, _box)
            nv_nl, nv_ptr, nv_mask, nv_ushift = nl_state['fn'](_pos, _cell, return_unit_shifts=True)
            nv_senders, nv_receivers = nv_nl
            per_atom_counts = nv_ptr[1:] - nv_ptr[:-1]
            max_per_atom = jnp.max(per_atom_counts)
            near = max_per_atom >= jnp.int32(int(nl_state['K'] * 0.9))

            def _check_overflow(max_count):
                K = nl_state['K']
                if int(max_count) >= K:
                    raise RuntimeError(
                        f"nvalchemi NL per-atom overflow: max {int(max_count)} neighbors "
                        f"for one atom >= per-atom capacity {K}. "
                        f"Edges are being silently dropped. "
                        f"Re-initialize the neighbor list (reduce density or increase "
                        f"neighbors_per_atom_safety in estimate_inputs_to_neighbor_list)."
                    )
            jax.debug.callback(_check_overflow, max_per_atom)

            return NvalchemiNbrs(
                reference_position=new_positions,
                did_buffer_overflow=near,
                senders=nv_senders,
                receivers=nv_receivers,
                edge_mask=nv_mask,
                unit_shifts=nv_ushift,
                ptr=nv_ptr,
            )

        @chex.dataclass
        class NvalchemiNbrs:
            """Neighbor list object for nvalchemi carrying pre-computed NL outputs.

            Storing NL arrays (shape M) here means a budget regrowth changes
            pytree leaf shapes, which triggers automatic JIT retracing of any
            downstream function (energy_fn, force grad, MD step).
            """
            reference_position: Array
            did_buffer_overflow: Array
            senders: Array
            receivers: Array
            edge_mask: Array
            unit_shifts: Array
            ptr: Array

            def update(self, new_positions, **kwargs):
                return _nvalchemi_update(new_positions, kwargs.get('box'))

        class NvalchemiNeighborFn:
            """Neighbor function with adaptive budget re-estimation on allocate."""

            @property
            def current_budget(self):
                """Return (K, M) — the current per-atom and global edge budgets."""
                return nl_state['K'], nl_state['M']

            def allocate(self, positions, box=None, **kwargs):
                import math
                _box = box if box is not None else nl_state['current_box']
                if _box is None:
                    raise ValueError(
                        "MACE (nvalchemi) potentials require a periodic box. "
                        "Non-periodic systems (box=None) are not supported."
                    )
                nl_state['current_box'] = _box
                _pos_f32 = jnp.asarray(positions, dtype=jnp.float32)
                _cell_f32 = _box.T.astype(jnp.float32)
                # Wrap before estimating so nvalchemi sees in-box positions.
                _pos_f32 = _wrap_to_box_f32(_pos_f32, _box)
                M_est, K_est, shifts_new = estimate_inputs_to_neighbor_list(
                    _pos_f32, nl_r_cutoff, cell=_cell_f32, assume_mic=False,
                )
                K_old, M_old = nl_state['K'], nl_state['M']
                # Only grow K/M when the estimate exceeds the current budget.
                # Unconditional growth (K_new = max(K_est, K_old * 1.2)) causes
                # shapes to diverge across sequential per-replica allocation
                # loops: each call inflates K by 20 %, so replica i gets M_i
                # and replica i+1 gets M_{i+1} > M_i, breaking jnp.stack.
                # The canonical case is "restore best-force positions" after an
                # L-BFGS explosion — K balloons to 4000+ for extreme positions,
                # then allocate() is called on the safe best positions (K~300)
                # 16 times in a row, each time growing K further for no reason.
                if int(K_est) > K_old or int(M_est) > M_old:
                    K_new = max(int(K_est), math.ceil(K_old * 1.2))
                    M_new = max(int(M_est), math.ceil(M_old * 1.2))
                    nl_state['K'] = K_new
                    nl_state['M'] = M_new
                    nl_state['shifts'] = shifts_new
                    # Re-estimate max_total_cells for cell_list backend when box changes.
                    if nl_state['algo'] == 'cell_list':
                        nl_state['max_total_cells'] = estimate_cell_list_params(
                            _pos_f32, nl_r_cutoff, _cell_f32
                        )
                    nl_state['fn'] = partial(
                        neighbor_list_jax_jit,
                        shifts=shifts_new,
                        cutoff=nl_r_cutoff,
                        max_neighbors_per_atom=K_new,
                        max_neighbors=M_new,
                        fill_value=0,
                        algorithm=nl_state['algo'],
                        max_total_cells=nl_state['max_total_cells'],
                    )
                    print(f'  NL reallocated: K {K_old}→{K_new}, M {M_old}→{M_new}')
                return _nvalchemi_update(positions, box)

            def update(self, positions, neighbor=None, **kwargs):
                return _nvalchemi_update(positions, kwargs.get('box'))

        neighbor_fn = NvalchemiNeighborFn()

        def energy_fn(positions, *, neighbor, lambda_val=None, lambda_electrostatics=None,
                      decouple_indices=None, total_charge_a=None, total_spin_a=None,
                      total_charge_b=None, total_spin_b=None, **kwargs):
            """Compute MACE energy using nvalchemi neighbor list.

            If a delta model is configured, returns base_energy + delta_energy.
            The supplied `neighbor` (NvalchemiNbrs) must be aligned to the current
            (positions, box); the runner ensures this via update_neighbor_list, and
            the MC barostat path uses a wrapper that refreshes nbrs before each call.
            """
            n = positions.shape[0]
            current_box = kwargs.get('box', box_for_nl)
            current_cell = current_box.T

            # Consume NL outputs from the supplied neighbor object directly.
            # dE/dbox flows through the live current_cell multiply on nv_out_shifts.
            nv_senders = neighbor.senders
            nv_receivers = neighbor.receivers
            nv_edge_mask = neighbor.edge_mask
            nv_unit_shifts = jax.lax.stop_gradient(neighbor.unit_shifts)
            nv_out_shifts = -nv_unit_shifts.astype(current_cell.dtype) @ current_cell

            # Process edges — positions_wrapped_f32 must NOT be stop-gradient'd so that
            # forces (dE/dpositions) flow correctly through the pair vector computation.
            positions_f32 = positions.astype(jnp.float32)
            positions_wrapped_f32 = _wrap_to_box_f32(
                positions_f32, jax.lax.stop_gradient(current_box)
            )

            nv_senders_safe = jnp.clip(nv_senders, 0, n - 1)
            nv_receivers_safe = jnp.clip(nv_receivers, 0, n - 1)
            nv_vectors = (positions_wrapped_f32[nv_receivers_safe] - positions_wrapped_f32[nv_senders_safe]) + nv_out_shifts

            valid_senders = jnp.where(nv_edge_mask, nv_senders, n)
            valid_receivers = jnp.where(nv_edge_mask, nv_receivers, n)
            valid_vectors = jnp.where(nv_edge_mask[:, None], nv_vectors, jnp.zeros_like(nv_vectors))

            edges_sorted, segments, vectors_sorted = sort_edges_by_receiver(
                edges=(valid_senders, valid_receivers),
                N=n,
                vectors=valid_vectors,
            )

            senders_sorted, receivers_sorted = edges_sorted
            edge_mask = (senders_sorted < n) & (receivers_sorted < n)

            senders_safe = jnp.clip(senders_sorted, 0, n - 1)
            receivers_safe = jnp.clip(receivers_sorted, 0, n - 1)
            naive_vectors = positions_f32[receivers_safe] - positions_f32[senders_safe]
            periodic_shifts = vectors_sorted - naive_vectors

            periodic_shifts = jnp.where(
                edge_mask[:, None],
                periodic_shifts,
                jnp.zeros_like(periodic_shifts),
            )

            # Build base model kwargs
            model_kwargs = dict(
                positions=positions,
                node_species=species_indices,
                edges_sorted_by_receiver=edges_sorted,
                segments=segments,
                periodic_shifts=periodic_shifts,
                edge_mask=edge_mask,
            )
            if lambda_val is not None:
                model_kwargs['lmbda'] = lambda_val
            if decouple_indices is not None:
                model_kwargs['decouple_indices'] = decouple_indices

            output = model.apply(params, **model_kwargs)
            total_energy = output.total_energy

            # Add delta model energy if configured
            if delta_model is not None:
                delta_kwargs = dict(
                    positions=positions,
                    node_species=delta_species_indices,
                    edges_sorted_by_receiver=edges_sorted,
                    segments=segments,
                    periodic_shifts=periodic_shifts,
                    edge_mask=edge_mask,
                )
                if lambda_val is not None:
                    delta_kwargs['lmbda'] = lambda_val
                if decouple_indices is not None:
                    delta_kwargs['decouple_indices'] = decouple_indices
                delta_output = delta_model.apply(delta_params, **delta_kwargs)
                total_energy = total_energy + delta_output.total_energy

            return total_energy

        return neighbor_fn, energy_fn, params

    elif model_type == 'mace_polar':
        # PolarMACE with electrostatics (Fukui loop)
        from aai_mace.mace_jax.save_load_utils import load_polar_mace_flax_from_orbax
        from aai_mace.mace_jax.electrostatics.fukui import (
            ElectrostaticGraph,
            get_jittable_electrostatic_graph_builder,
        )
        from pathlib import Path
        from jax_md import space

        save_dir = Path(checkpoint_path).expanduser().resolve()
        
        # Load model using the save_load_utils function.
        # Pass the simulation dtype so float64 checkpoints load as float64.
        # For float64 mode, also override tp_precision and contraction_precision to HIGHEST
        # so the cuEquivariance einsum uses float64 accumulators (matches MACE-OFF behavior).
        # PolarMACE defaults to "float32" for these, which causes float64→float32 scatter NaN.
        sim_dtype = positions.dtype
        _precision_kwargs = {}
        if sim_dtype == jnp.float64:
            _precision_kwargs = {
                'tp_precision': jax.lax.Precision.HIGHEST,
                'contraction_precision': jax.lax.Precision.HIGHEST,
            }
        model_apply, params, atomic_numbers_in_model, model_config, extra_config = load_polar_mace_flax_from_orbax(
            str(save_dir), dtype=sim_dtype, **_precision_kwargs
        )
        
        # Replicate params across mesh if requested (for shard_map parallelism)
        if mesh_for_replication is not None:
            print(f'Replicating PolarMACE params across mesh: {mesh_for_replication.shape}')
            params = replicate_params_to_mesh(params, mesh_for_replication)
        
        # Get r_max from model config - this MUST match the training cutoff
        if 'r_max' not in model_config:
            raise ValueError(
                f"Model config missing 'r_max'. Cannot determine neighbor list cutoff. "
                f"Checkpoint: {save_dir}"
            )
        r_cutoff = model_config['r_max']
        print(f'Using PolarMACE model r_max = {r_cutoff} Å for neighbor list cutoff')
        
        # Get kspace_cutoff from extra_config or use config override
        kspace_cutoff = config.kspace_cutoff
        if kspace_cutoff is None:
            kspace_cutoff = extra_config.get('kspace_cutoff', r_cutoff)
        print(f'Using kspace_cutoff = {kspace_cutoff} Å for electrostatics')
        
        # Get total_charge and total_spin from config
        total_charge = config.total_charge
        total_spin = config.total_spin
        print(f'Using total_charge = {total_charge}, total_spin = {total_spin}')
        
        # Convert atomic numbers to species indices based on atomic_numbers from model
        if atomic_numbers_in_model is not None:
            atomic_num_to_species_idx = {int(z): i for i, z in enumerate(atomic_numbers_in_model)}
            species_indices = jnp.array([atomic_num_to_species_idx[int(z)] for z in species], dtype=jnp.int32)
        else:
            # If no atomic_numbers, assume atomic numbers map directly to indices
            species_indices = jnp.array([int(z) for z in species], dtype=jnp.int32)

        # Load optional delta MACE-OFF model
        delta_model = None
        delta_params = None
        delta_species_indices = None
        nl_r_cutoff = r_cutoff  # Cutoff used for the shared neighbor list
        if config.delta_checkpoint_path:
            print(f'Loading delta MACE-OFF model from {config.delta_checkpoint_path}')
            delta_model, delta_params, delta_species_indices, delta_r_cutoff, delta_cfg = (
                _load_mace_off_checkpoint(
                    config.delta_checkpoint_path, species, mesh_for_replication
                )
            )
            print(f'Delta model r_max = {delta_r_cutoff} Å, '
                  f'avg_num_neighbors = {delta_cfg["avg_num_neighbors"]:.2f}')
            nl_r_cutoff = max(r_cutoff, delta_r_cutoff)
            if nl_r_cutoff > r_cutoff:
                print(f'Using delta r_max = {nl_r_cutoff} Å for shared neighbor list '
                      f'(larger than base r_max = {r_cutoff} Å)')

        # =====================================================================
        # Create ElectrostaticGraph builder for k-space electrostatics
        # This needs to be done outside the energy_fn to capture the grid limits
        # =====================================================================
        box_for_nl = box if box is not None else jnp.eye(3)
        
        # Convert jax-md box to ASE cell convention (lower triangular, row vectors)
        # jax-md box has lattice vectors as columns, ASE has them as rows.
        # Keep float32 for nvalchemi NL (requires f32); use sim_dtype for kspace.
        cell_f32_for_nl = box_for_nl.T.astype(jnp.float32)
        example_cell = box_for_nl.T.astype(sim_dtype)  # kspace uses simulation dtype

        # Build the electrostatic graph builder (captures max_k_vectors and grid_limits)
        electrostatic_graph_builder = get_jittable_electrostatic_graph_builder(
            example_cell=example_cell,
            cutoff=kspace_cutoff,
            safety_factor=1.15,
            non_pbc_corrections=False,  # True for non-periodic systems
        )

        # =====================================================================
        # Neighbor list setup (nvalchemi only)
        # =====================================================================
        from functools import partial
        from jax_md_cli.nvalchemi_nl import (
            estimate_inputs_to_neighbor_list,
            neighbor_list_jax_jit,
            estimate_cell_list_params,
        )
        print(f'Using nvalchemi neighbor list backend for PolarMACE')

        # Resolve NL algorithm: 'auto' picks cell_list for large systems.
        algo = neighbor_config.algorithm
        if algo == 'auto':
            algo = (
                'cell_list'
                if positions.shape[0] > neighbor_config.algorithm_auto_threshold
                else 'naive'
            )
        print(f'  nvalchemi NL algorithm: {algo}')

        # Cell must be lower triangular for the nvalchemi NL (always float32)
        cell_for_nl = cell_f32_for_nl

        # Estimate neighbor list parameters from initial positions
        positions_f32 = positions.astype(jnp.float32)

        max_neighbors, max_neighbors_per_atom, shifts = estimate_inputs_to_neighbor_list(
            positions_f32,
            nl_r_cutoff,
            cell=cell_for_nl,
            assume_mic=False,
        )
        max_neighbors = 150000
        max_neighbors_per_atom = 500
        max_neighbors = positions.shape[0]*max_neighbors_per_atom
        print(f'  max_neighbors={max_neighbors}, max_neighbors_per_atom={max_neighbors_per_atom}')

        max_total_cells = None
        if algo == 'cell_list':
            max_total_cells = estimate_cell_list_params(positions_f32, nl_r_cutoff, cell_for_nl)
            print(f'  max_total_cells={max_total_cells}')

        # Mutable state for the NL budget; updated by NvalchemiNeighborFn.allocate().
        nl_state = {
            'fn': partial(
                neighbor_list_jax_jit,
                shifts=shifts,
                cutoff=nl_r_cutoff,
                max_neighbors_per_atom=max_neighbors_per_atom,
                max_neighbors=max_neighbors,
                fill_value=0,
                algorithm=algo,
                max_total_cells=max_total_cells,
            ),
            'K': max_neighbors_per_atom,
            'M': max_neighbors,
            'shifts': shifts,
            'algo': algo,
            'max_total_cells': max_total_cells,
            'current_box': box_for_nl,
        }

        def _wrap_to_box_f32(pos_f32, box):
            """Wrap f32 Cartesian positions to [0, box] via integer cell shifts.

            box has COLUMNS as box vectors (JAX-MD convention: box = cell_ase.T).
            pos = frac @ box.T so frac = pos @ inv(box.T).  Gradient w.r.t.
            pos_f32 is the identity (floor is piecewise-constant).  Pass
            stop_gradient(box) if box gradients must not flow through wrapping.
            """
            box_f32 = jnp.asarray(box, dtype=jnp.float32)
            # pos = frac @ box.T → frac = pos @ inv(box.T); pos_wrapped = (frac%1) @ box.T
            inv_boxT = jnp.linalg.inv(box_f32.T)
            frac = pos_f32 @ inv_boxT
            return pos_f32 - jnp.floor(frac) @ box_f32.T

        def _nvalchemi_update(new_positions, box=None):
            """Run NL and return NvalchemiNbrs with preventive overflow flag.

            Positions are wrapped to [0, box] before passing to nvalchemi so
            that the returned unit_shifts are consistent with _wrap_to_box_f32
            used in energy_fn.  Sets did_buffer_overflow when any atom uses
            >= 90% of its per-atom budget (issue #13 A).
            """
            _box = box if box is not None else nl_state['current_box']
            if _box is None:
                raise ValueError(
                    "MACE (nvalchemi) potentials require a periodic box. "
                    "Non-periodic systems (box=None) are not supported."
                )
            nl_state['current_box'] = _box
            _cell = jax.lax.stop_gradient(_box.T.astype(jnp.float32))
            _pos = jax.lax.stop_gradient(new_positions.astype(jnp.float32))
            _pos = _wrap_to_box_f32(_pos, _box)
            nv_nl, nv_ptr, nv_mask, nv_ushift = nl_state['fn'](_pos, _cell, return_unit_shifts=True)
            nv_senders, nv_receivers = nv_nl
            per_atom_counts = nv_ptr[1:] - nv_ptr[:-1]
            max_per_atom = jnp.max(per_atom_counts)
            near = max_per_atom >= jnp.int32(int(nl_state['K'] * 0.9))

            def _check_overflow(max_count):
                K = nl_state['K']
                if int(max_count) >= K:
                    raise RuntimeError(
                        f"nvalchemi NL per-atom overflow: max {int(max_count)} neighbors "
                        f"for one atom >= per-atom capacity {K}. "
                        f"Edges are being silently dropped. "
                        f"Re-initialize the neighbor list (reduce density or increase "
                        f"neighbors_per_atom_safety in estimate_inputs_to_neighbor_list)."
                    )
            jax.debug.callback(_check_overflow, max_per_atom)

            return NvalchemiNbrs(
                reference_position=new_positions,
                did_buffer_overflow=near,
                senders=nv_senders,
                receivers=nv_receivers,
                edge_mask=nv_mask,
                unit_shifts=nv_ushift,
                ptr=nv_ptr,
            )

        @chex.dataclass
        class NvalchemiNbrs:
            """Neighbor list object for nvalchemi carrying pre-computed NL outputs.

            Storing NL arrays (shape M) here means a budget regrowth changes
            pytree leaf shapes, triggering automatic JIT retracing of downstream
            functions (energy_fn, force grad, MD step).
            """
            reference_position: Array
            did_buffer_overflow: Array
            senders: Array
            receivers: Array
            edge_mask: Array
            unit_shifts: Array
            ptr: Array

            def update(self, new_positions, **kwargs):
                return _nvalchemi_update(new_positions, kwargs.get('box'))

        class NvalchemiNeighborFn:
            """Neighbor function with adaptive budget re-estimation on allocate."""

            @property
            def current_budget(self):
                """Return (K, M) — the current per-atom and global edge budgets."""
                return nl_state['K'], nl_state['M']

            def allocate(self, positions, box=None, **kwargs):
                import math
                _box = box if box is not None else nl_state['current_box']
                if _box is None:
                    raise ValueError(
                        "MACE (nvalchemi) potentials require a periodic box. "
                        "Non-periodic systems (box=None) are not supported."
                    )
                nl_state['current_box'] = _box
                _pos_f32 = jnp.asarray(positions, dtype=jnp.float32)
                _cell_f32 = _box.T.astype(jnp.float32)
                # Wrap before estimating so nvalchemi sees in-box positions.
                _pos_f32 = _wrap_to_box_f32(_pos_f32, _box)
                M_est, K_est, shifts_new = estimate_inputs_to_neighbor_list(
                    _pos_f32, nl_r_cutoff, cell=_cell_f32, assume_mic=False,
                )
                K_old, M_old = nl_state['K'], nl_state['M']
                # Only grow K/M when the estimate exceeds the current budget.
                # Unconditional growth (K_new = max(K_est, K_old * 1.2)) causes
                # shapes to diverge across sequential per-replica allocation
                # loops: each call inflates K by 20 %, so replica i gets M_i
                # and replica i+1 gets M_{i+1} > M_i, breaking jnp.stack.
                # The canonical case is "restore best-force positions" after an
                # L-BFGS explosion — K balloons to 4000+ for extreme positions,
                # then allocate() is called on the safe best positions (K~300)
                # 16 times in a row, each time growing K further for no reason.
                if int(K_est) > K_old or int(M_est) > M_old:
                    K_new = max(int(K_est), math.ceil(K_old * 1.2))
                    M_new = max(int(M_est), math.ceil(M_old * 1.2))
                    nl_state['K'] = K_new
                    nl_state['M'] = M_new
                    nl_state['shifts'] = shifts_new
                    # Re-estimate max_total_cells for cell_list backend when box changes.
                    if nl_state['algo'] == 'cell_list':
                        nl_state['max_total_cells'] = estimate_cell_list_params(
                            _pos_f32, nl_r_cutoff, _cell_f32
                        )
                    nl_state['fn'] = partial(
                        neighbor_list_jax_jit,
                        shifts=shifts_new,
                        cutoff=nl_r_cutoff,
                        max_neighbors_per_atom=K_new,
                        max_neighbors=M_new,
                        fill_value=0,
                        algorithm=nl_state['algo'],
                        max_total_cells=nl_state['max_total_cells'],
                    )
                    print(f'  NL reallocated: K {K_old}→{K_new}, M {M_old}→{M_new}')
                return _nvalchemi_update(positions, box)

            def update(self, positions, neighbor=None, **kwargs):
                return _nvalchemi_update(positions, kwargs.get('box'))

        neighbor_fn = NvalchemiNeighborFn()

        def energy_fn(positions, *, neighbor, lambda_val=None, lambda_electrostatics=None,
                      decouple_indices=None, total_charge_a=None, total_spin_a=None,
                      total_charge_b=None, total_spin_b=None, **kwargs):
            """Compute PolarMACE energy using nvalchemi neighbor list.

            The supplied `neighbor` (NvalchemiNbrs) must be aligned to the current
            (positions, box); the runner ensures this via update_neighbor_list, and
            the MC barostat path uses a wrapper that refreshes nbrs before each call.

            Args:
                positions: Cartesian positions [N, 3]
                neighbor: NvalchemiNbrs carrying the current NL outputs
                lambda_val: Optional lambda for local (short-range) alchemical interactions [0,1]
                lambda_electrostatics: Optional lambda for electrostatic scaling [0,1]
                    If None, defaults to lambda_val (same scaling for both)
                decouple_indices: Optional atom indices for alchemical decoupling
                total_charge_a: Optional charge of fragment A (non-decoupled atoms) at lambda=0
                total_spin_a: Optional spin of fragment A at lambda=0
                total_charge_b: Optional charge of fragment B (decoupled atoms) at lambda=0
                total_spin_b: Optional spin of fragment B at lambda=0
                **kwargs: Must include 'box' for periodic systems
            """
            n = positions.shape[0]
            current_box = kwargs.get('box', box_for_nl)
            current_cell = current_box.T

            # Build ElectrostaticGraph using the simulation dtype for the cell.
            # kspace.py already casts to the input cell's dtype, so float64 works.
            electrostatic_graph, truncated = electrostatic_graph_builder(current_cell.astype(sim_dtype))

            # Consume NL outputs from the supplied neighbor object directly.
            # dE/dbox flows through the live current_cell multiply on nv_out_shifts.
            nv_senders = neighbor.senders
            nv_receivers = neighbor.receivers
            nv_edge_mask = neighbor.edge_mask
            nv_unit_shifts = jax.lax.stop_gradient(neighbor.unit_shifts)
            nv_out_shifts = -nv_unit_shifts.astype(current_cell.dtype) @ current_cell

            # Gradient path for pair vectors: use the simulation dtype (sim_dtype)
            # so forces and kspace remain in float64 when requested.
            # Wrapping: floor is non-differentiable, so stop-gradient the shift;
            # gradient flows through positions only (box is treated as fixed here).
            _inv_boxT_sg = jax.lax.stop_gradient(
                jnp.linalg.inv(current_box.T).astype(sim_dtype)
            )
            _frac = positions.astype(sim_dtype) @ _inv_boxT_sg
            positions_wrapped = (
                positions.astype(sim_dtype)
                - jax.lax.stop_gradient(jnp.floor(_frac))
                @ jax.lax.stop_gradient(current_box.T.astype(sim_dtype))
            )

            # Compute vectors using nvalchemi's shifts (already in simulation dtype
            # via the current_cell.dtype cast of nv_out_shifts above).
            nv_senders_safe = jnp.clip(nv_senders, 0, n - 1)
            nv_receivers_safe = jnp.clip(nv_receivers, 0, n - 1)
            nv_vectors = (positions_wrapped[nv_receivers_safe] - positions_wrapped[nv_senders_safe]) + nv_out_shifts

            # Filter to valid edges
            valid_senders = jnp.where(nv_edge_mask, nv_senders, n)
            valid_receivers = jnp.where(nv_edge_mask, nv_receivers, n)
            valid_vectors = jnp.where(nv_edge_mask[:, None], nv_vectors, jnp.zeros_like(nv_vectors))

            # Sort edges and vectors together by receiver
            edges_sorted, segments, vectors_sorted = sort_edges_by_receiver(
                edges=(valid_senders, valid_receivers),
                N=n,
                vectors=valid_vectors,
            )

            senders_sorted, receivers_sorted = edges_sorted
            edge_mask = (senders_sorted < n) & (receivers_sorted < n)

            # Compute periodic_shifts for MACE.
            # naive_vectors uses the same positions the model receives so the
            # identity positions[r] - positions[s] + periodic_shifts = min-image holds.
            senders_safe = jnp.clip(senders_sorted, 0, n - 1)
            receivers_safe = jnp.clip(receivers_sorted, 0, n - 1)
            naive_vectors = positions.astype(sim_dtype)[receivers_safe] - positions.astype(sim_dtype)[senders_safe]
            periodic_shifts = vectors_sorted - naive_vectors

            periodic_shifts = jnp.where(
                edge_mask[:, None],
                periodic_shifts,
                jnp.zeros_like(periodic_shifts),
            )
            
            # Build model kwargs for PolarMACE
            model_kwargs = dict(
                positions=positions,
                node_species=species_indices,
                edges_sorted_by_receiver=edges_sorted,
                electrostatic_graph=electrostatic_graph,
                segments=segments,
                total_charge=total_charge,
                total_spin=total_spin,
                external_potential=None,
                periodic_shifts=periodic_shifts,
                edge_mask=edge_mask,  # Critical: mask invalid padded edges
            )
            
            # Add alchemical parameters if provided
            if lambda_val is not None:
                model_kwargs['lmbda'] = lambda_val
            if decouple_indices is not None:
                model_kwargs['decouple_indices'] = decouple_indices
            if lambda_electrostatics is not None:
                model_kwargs['lmbda_electrostatics'] = lambda_electrostatics
            # Per-fragment charge/spin for alchemical normalization
            if total_charge_a is not None:
                model_kwargs['total_charge_a'] = total_charge_a
            if total_spin_a is not None:
                model_kwargs['total_spin_a'] = total_spin_a
            if total_charge_b is not None:
                model_kwargs['total_charge_b'] = total_charge_b
            if total_spin_b is not None:
                model_kwargs['total_spin_b'] = total_spin_b

            output = model_apply(params, **model_kwargs)
            total_energy = output.total_energy

            # Add delta model energy if configured
            if delta_model is not None:
                delta_kwargs = dict(
                    positions=positions,
                    node_species=delta_species_indices,
                    edges_sorted_by_receiver=edges_sorted,
                    segments=segments,
                    periodic_shifts=periodic_shifts,
                    edge_mask=edge_mask,
                )
                if lambda_val is not None:
                    delta_kwargs['lmbda'] = lambda_val
                if decouple_indices is not None:
                    delta_kwargs['decouple_indices'] = decouple_indices
                delta_output = delta_model.apply(delta_params, **delta_kwargs)
                total_energy = total_energy + delta_output.total_energy

            return total_energy

        return neighbor_fn, energy_fn, params

    else:
        raise ValueError(f'Unknown model type: {model_type}')


def build_classical_potential(
    config: ClassicalPotentialConfig,
    displacement_fn: Callable,
    box: Array,
    neighbor_config: NeighborListConfig,
    species: Optional[Array] = None,
    fractional_coordinates: bool = True,
) -> Tuple[Callable, Callable]:
    """Build classical potential energy function.

    Args:
      config: Classical potential configuration.
      displacement_fn: Displacement function from space module.
      box: Simulation box.
      neighbor_config: Neighbor list configuration.
      species: Species indices for multi-component systems.
      fractional_coordinates: Whether using fractional coordinates.

    Returns:
      neighbor_fn: Function to create/update neighbor lists.
      energy_fn: Energy function.
    """
    ptype = config.type
    nl_format = get_neighbor_list_format(neighbor_config.format)

    # Convert sigma/epsilon to arrays if needed
    sigma = jnp.array(config.sigma)
    epsilon = jnp.array(config.epsilon)

    if ptype == 'lennard_jones':
        neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
            displacement_fn,
            box if box is not None else jnp.zeros((3,3)),
            species=species,
            sigma=sigma,
            epsilon=epsilon,
            r_onset=config.r_onset,
            r_cutoff=config.r_cutoff,
            dr_threshold=neighbor_config.dr_threshold,
            fractional_coordinates=fractional_coordinates,
            format=nl_format,
            capacity_multiplier=neighbor_config.capacity_multiplier,
        )

    elif ptype == 'soft_sphere':
        neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
            displacement_fn,
            box,
            species=species,
            sigma=sigma,
            epsilon=epsilon,
            alpha=jnp.array(config.alpha),
            dr_threshold=neighbor_config.dr_threshold,
            fractional_coordinates=fractional_coordinates,
            format=nl_format,
            capacity_multiplier=neighbor_config.capacity_multiplier,
        )

    elif ptype == 'morse':
        # Morse potential with neighbor list
        neighbor_fn, energy_fn = energy.morse_neighbor_list(
            displacement_fn,
            box,
            species=species,
            sigma=sigma,
            epsilon=epsilon,
            alpha=jnp.array(config.alpha),
            r_onset=config.r_onset,
            r_cutoff=config.r_cutoff,
            dr_threshold=neighbor_config.dr_threshold,
            fractional_coordinates=fractional_coordinates,
            format=nl_format,
            capacity_multiplier=neighbor_config.capacity_multiplier,
        )

    else:
        raise ValueError(f'Unknown potential type: {ptype}')

    return neighbor_fn, energy_fn