# Predicting Lower Limb Muscle Forces from Ground Reaction Forces During Gait Using Sequence and Attention-Based Deep Learning Models

## Description
This readme describes the context into which the demo script fits and how to run the demo script to test the models on a sample input of ground reaction force data. To understand how to retrain the models please refer to the other README files.

The goal of this project was to develop deep learning models that can predict 11 major ankle-crossing muscles using only ground reaction forces as inputs. These models were trained on a combination of younger adult and older adult individuals. There were 10  healthy younger adult individuals walking on a force-instrumented treadmill in a motion capture labratory that has been collected, processed, and open-sourced by [Uhlrich et al., 2022](https://doi.org/10.1038/s41598-022-13386-9). There were 20 healthy older adult individuals walking on 3 discretized force plates in a motion capture labratory that were collected by [Silder et al., 2008](https://doi.org/10.1016/j.jbiomech.2008.02.016). 

The older adult data was preprocessed through scaling and rotation to format in a proper format for OpenSim processing. Inverse kinematics and inverse dynamics were run through opensim scripting, and a custom static optimization script from [Uhlrich et al., 2022](https://doi.org/10.1038/s41598-022-13386-9) were used to estimate ground truth muscle forces to train the models.

Four models were explored, including LSTM, CNN-LSTM, LSTM with Attention, and a Transformer. All models achieved strong predictive performance, and the LSTM and Transformer performed the best. These results highlight the potential of deep learning to capture the complex relationships and patterns between GRFs and below knee muscle forces during gait. This work provides groundwork for expanding access to musculoskeletal analysis using solely ground reaction forces without kinematics being captured.

A demo video describing the process and results from the full training of the models can be found [here](insert link)


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

1. All data for demo script is found in the "Demo_data" folder.

### Execution

Demo script describes steps for execution. Ensure it is in the 'OA_script' directory, which is a subdirectory of the main repository. The demo data should be in the root folder, and is referenced as such in the demo script. 

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
* Brian Keller, brian.keller@utah.edu
* Ty Lunde, u0718706@utah.edu

## Verison History

* 0.1
  * Initial Release

## Acknowledgements

1. Uhlrich SD, Jackson RW, Seth A, Kolesar JA, Delp SL, 2022. Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. Scientific Reports 12, 9842. https://doi.org/10.1038/s41598-022-13386-9.
2. Silder, A., Heiderscheit, B. and Thelen, D.G., 2008. Active and passive contributions to joint kinetics during walking in older adults. *Journal of biomechanics*, 41(7), pp.1520-1527. https://doi.org/10.1016/j.jbiomech.2008.02.016.
