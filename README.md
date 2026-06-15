## Overview
The goal of this project is to investigate solubility of NH3 and H2S in superionized water, with application to the mystery of excess free H2S clouds in Uranus's atmosphere, where solar abudance ratios would imply all H2S reacts with excess NH3 to form NH4SH.

Plan:

1. Get practice with MD and MLIP (Machine Learned Interatomic Potentials) by fine tuning MACE-POLAR with Binqqing Chen's water data (repo is called ab-initio-thermodynamics-of-water)

2. Perform DFT simulations of aqueous NH3 and H2S mixtures to fine tune MLIP

3. Use alchemical free energy methodology (gradually turn on water-solute interactions) to determine free energy of mixing and solubility for NH3 and H2S respectively

Some useful reading is in the "reading/" folder

## Journal
Monday June 16th
 - Organized work into one github repo, interpretable to codex agent
 - so far, able to run naive fine tuning of mace-polar and run molecular dynamics simulations of water
 - Understand energy reference of training data (isolated atom energies not available, -E0 = "estimate" setting better than "average" for fine tuning)

Next Steps:
 - Understand how to use autocorrelation to determine errors
  - Understand how to use multihead fine tuning rather than naive fine tuning
   - familiarize with DFT, Omol25 dataset,
https://orca-manual.mpi-muelheim.mpg.de/contents/modelchemistries/DensityFunctionalTheory.html,
https://www.faccts.de/orca/

   - Want to run DFT of NH3, H2O, H2S over weekend, need to familiarize with rest of pipeloine



## Dependencies

This project depends on the following external repositories, which should be cloned separately into the project root:

```bash
git clone https://github.com/ACEsuit/mace.git
git clone https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water.git
git clone https://github.com/imagdau/aseMolec.git