# Predicting Lower Limb Muscle Forces from Ground Reaction Forces During Gait Using Sequence and Attention-Based Deep Learning Models

## Description

This project aims to develop a data-driven model  of the relationship between ground reaction forces and muscle forces using deep learning techniques. This model will be trained on a data from ten healthy individuals walking on a force-instrumented treadmill in a motion capture laboratory that has been collected, processed, and open-sourced by [Uhlrich et al., 2022](https://doi.org/10.1038/s41598-022-13386-9).

Four models were explored, including LSTM, CNN-LSTM, LSTM with Attention, and a Transformer. All models achieved strong predictive performance, with the Transformer model consistently outperforming others in accuracy across most muscle forces and overall. These results highlight the potential of deep learning to capture the complex relationships and patterns between GRFs and lower limb muscle forces during gait. This work provides the groundwork for advancing data-driven approaches in robotic cadaveric gait simulation, enabling more reliable and flexible control strategies to replicate physiological motions in cadavers.

## Getting Started

### Dependencies

* Python 3.11.10
* PyTorch 2.5.1 or later
* NumPy
* SciPy
* Matplotlib
* OpenSim 4.5 or later
* OpenSim Python API
* OpenSim MATLAB API

### Data

1. Download dataset from the [Coordination Retraining Project](https://simtk.org/projects/coordretraining) on SimTK
2. Extract the dataset to the 'data' directory

### Additional Documentation

* [OpenSim API Documentation](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Contributing

1. Create your branch
2. Commit your changes
3. Push to the branch
4. Open a pull request

## License

This project is licensed under the Creative Commons Zero License.

## Authors

* Anthony Le ([anth0nyhle](https://github.com/anth0nyhle), anthony.le@utah.edu)

## Verison History

* 0.1
  * Initial Release

## Acknowledgements

1. Uhlrich SD, Jackson RW, Seth A, Kolesar JA, Delp SL, 2022. Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. Scientific Reports 12, 9842. https://doi.org/10.1038/s41598-022-13386-9.
