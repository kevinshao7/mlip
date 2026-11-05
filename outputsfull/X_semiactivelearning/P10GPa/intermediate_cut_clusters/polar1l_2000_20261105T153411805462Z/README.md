# P10GPa intermediate cut clusters: polar-1-l, 2000 geometries

`clusters_2000.xyz` is extended XYZ: 1950 acquisition records first, then 50 evaluation records.
Keep `cluster_kind=evaluation` out of training. Source frame/replica, charge and multiplicity
are retained in every record and in `clusters_2000_manifest.json`. No DFT labels are present.

Source: ten independent GPU MD runs, 10 GPa, 1920 K, 0.5 fs, periodic MTK NPT;
91–92.5 ps partial production, cut from 10.5 ps onward. Each isolated cluster has
six whole molecules. Nearby frames/centres are correlated; no equilibrium claim.

`provenance/` contains immutable historical records; absolute source paths there
identify the original machine and are not required to read this portable dataset.
`checksums.json` covers every file in this package except itself. Raw trajectories,
model weights and restarts are deliberately outside the Git package.
