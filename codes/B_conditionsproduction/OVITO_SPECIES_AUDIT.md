# Checking species assignments in OVITO

Open `speciestest.xyz` in OVITO and accept the detected **Extended XYZ** format.
The file contains at most 20 animation frames, focused on species categorized
by the time-series plot as `unexpected_cluster` or `improper_species`. Sampling
is balanced across the selected runs. Each frame contains one species under
test plus its four nearest complete species. The atoms have these imported
particle properties:

- `molecule`: string formula of the molecular component that the atom was
  assigned to, such as `H5N`, `H2O`, or `O`.
- `component_id = 0`: species under test; values 1--4 identify its neighbors.
- `source_atom_id`: zero-based atom index in the original production frame.

The Data Inspector and frame attributes also show `target_species`, `target_category`, `run_id`,
`condensed_frame_index_0based`, `target_source_atom_ids`, and
`neighbor_species`.

## Highlight the target species

1. The target is still component zero, so add **Select expression** and enter
   `component_id == 0`.
2. Add **Assign color**, enable **Apply to selected particles only**, and choose
   a vivid color such as red.
3. Inspect the string `molecule` column in OVITO's Data Inspector to see the
   supposed species membership of every atom.
4. In **Particles** visual settings, increase the radius to make atoms easier
   to see. Step through animation frames; each frame audits a different formula.

To color all five components separately instead, add **Color coding**, choose
the `component_id` property, and use a categorical-looking color map.

## Point at one atom and inspect its molecule label

The quickest method does not require adding a modifier:

1. Expand the **Data Inspector** panel below the viewport and open its
   **Particles** tab.
2. Click **Select in viewports** (the crosshair/pick icon in the Particles
   toolbar).
3. Click an atom in the 3-D viewport. The selected atom is highlighted red and
   the particle table is filtered to its row. Hold **Ctrl** while clicking to
   select more atoms. Right-click the viewport, or click the crosshair again,
   to leave pick mode.
4. Read the selected row's `molecule` column. This is the string formula that
   the production analysis assigned to that atom. Also inspect
   `component_id`, `source_atom_id`, particle type, and position in the same
   row. `component_id = 0` denotes the species under test in that frame.

If the table is hidden, click its collapsed tab bar below the viewport. Drag
column boundaries or horizontally scroll if the `molecule` column is off-screen.

For a selection that stays in the modifier pipeline, add **Manual selection**,
activate its **Pick mode**, and click atoms. Use **Ctrl** to add and **Alt** to
remove atoms. Selected particles are highlighted red in the interactive
viewport. A subsequent **Assign color** modifier can give that selection a
permanent display color (and a rendered color, unlike the temporary red
selection highlight).

In OVITO 3.16 or newer, a string property can also be used by **Expression
selection**. For example, enter `molecule == "H5N"` to select every atom
assigned to an H5N component in the current audit frame. Use
`component_id == 0` when you want only the single target component, because
several neighboring components can have the same molecular formula.

## Show bonds without redefining the species

The file deliberately stores component membership rather than guessed bonds.
If desired, add OVITO's **Create bonds** modifier only as a visual aid. Do not
use the resulting OVITO bonds to decide membership: the plotted analysis assigns
every H unconditionally to its nearest H/O/N using periodic minimum-image
distance, and uses 1.20 times covalent radii only for heavy-heavy bonds.

Regenerate the audit with:

```powershell
python make_species_ovito_audit.py
```

By default this samples every 100 GPa run, matching the pressure filter in the
final-species plotting script, uses four parallel worker processes, and stops
after four qualifying examples per run.
Use repeated `--run-id` options to audit selected conditions instead, or change
the hard cap with `--max-examples`. Change CPU parallelism with `--workers`.
