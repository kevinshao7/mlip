# Campaign decisions record

This record preserves the user-approved scientific design. Change the JSON
configuration and append a provenance event before changing any item.

1. Campaign conditions are NH3/H2O R1, one random MD seed per condition, with
   pressure targets 10, 17.7828, 31.0, 56.2341, and 100 GPa. The 31 GPa
   condition is intentionally rounded rather than using 31.6228 GPa.
2. Use the current B-production simulation size, fixed Uranus profile policy,
   0.5 fs timestep, 37 ps production hold (82,000 total MD steps), an
   approximately five-hour runtime target per 350-atom replica, and 2 ps
   pressure plus 2 ps temperature ramps. Only production-hold frames are
   saved: a computed 37-step interval produces exactly 2,000 frames.
3. The component assignment is the B-production nearest-heavy-atom policy.
   Base species are only H2O, HO, H3O, NH3, NH2, and NH4 (stored by the
   formatter as H3N, H2N, and H4N). Any other complete component triggers
   human review and may supply an event-centred cluster.
4. Illegal species are diagnosis-only event markers and never enter DFT or
   training data. Every candidate DFT/training cluster contains the closest
   allowed central molecule plus its five nearest allowed complete molecular
   neighbours under PBC (six total).
5. The active-learning sweet spot requires at least one non-base component but
   an illegal complete-component fraction below 5%; higher fractions require
   human review rather than automatic cluster generation.
6. Generate at most 10 diagnostic DFT clusters first. Review species and parity
   plots manually; roughly 50 meV/atom energy error is guidance, never an
   automatic trigger. A recorded approval unlocks at most 50 acquisition DFT
   clusters.
7. ORCA is configured for 24 CPU ranks. Two GPUs are reserved operationally:
   MD defaults to GPU 0 and fine tuning to GPU 1; change only in the config.
8. Historical campaign runs began from `polar-1-s`; those artifacts remain
   immutable for reproducibility. As of 2026-10-28, `polar-1-s` is retired for
   new work. New campaign runs begin from `polar-1-l`. After an approved campaign
   fine tune, later campaign MD uses its direct child checkpoint. New validated
   DFT data accumulates into training while the D2 validation/test sets remain fixed.
