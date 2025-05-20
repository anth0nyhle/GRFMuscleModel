# Old Young Walking Data

## Overview

This dataset contains gait biomechanical data from both older and younger adult participants during overground working at various speeds.

Dataset location: L:\Project_Data\GRF_Muscle_Model\Old Young Walking Data

## Data Description

### Subjects

* Total: 40 participants
* Groups:
  * 20 healthy young (18-35 yrs old)
  * 20 healthy old (65-85 yrs old)

### Experimental Setup

* Environment: overground walking in lab
* Walking speeds:
  * Normalized speeds based on preferred walking speed
  * Speeds: 80%, 100%, 120% of preferred walking speed

### Collected Data

* File types:
  * .anc - analog data for force plate and muscle foce data
    * Force plate data (F1X, F1Y, F1Z, M1X, M1Y, M1Z, F2X, F2Y, F2Z, M2X, M2Y, M2Z, F3X, F3Y, F3Z, M3X, M3Y, M3Z)
    * Muscule forces (SOL, GAS, BF, MH, TA, VL, RF)
  * .anb - binary data (anything with "b")
  * .trc - marker data (X, Y, Z)
    * HJC trials - circumduction movement to determine hip joint center
    * R - right
    * L - left
    * SH - shank plate
    * TH - thigh plate
    * Other marker names are self-explanatory
  * .forces - not filtered, needed to recalculate COP
    * Force plate data (F1X, F1Y, F1Z, M1X, M1Y, M1Z, F2X, F2Y, F2Z, M2X, M2Y, M2Z, F3X, F3Y, F3Z, M3X, M3Y, M3Z)
    * Muscule forces (SOL, GAS, BF, MH, TA, VL, RF)
  * .static - static trials in T-pose
  * *ik.mot - inverse kinematics (probably from OpenSim?)
  * *id.mot - inverse dynamics (probably from OpenSim?)
* Files_W_HJCs - this folder might contain some processed data like the .mot files that are usually created from OpenSim -- explore and test

### Citation

Silder, A., Heiderscheit, B. and Thelen, D.G., 2008. Active and passive contributions to joint kinetics during walking in older adults. *Journal of biomechanics*, 41(7), pp.1520-1527.

``` bibtex
@article{silder2008active,
  title={Active and passive contributions to joint kinetics during walking in older adults},
  author={Silder, Amy and Heiderscheit, Bryan and Thelen, Darryl G},
  journal={Journal of biomechanics},
  volume={41},
  number={7},
  pages={1520--1527},
  year={2008},
  publisher={Elsevier}
}
```
