% Custom static optimization code. Author: Scott Uhlrich, Stanford
% University, 2020. Please cite:
% Uhlrich, S.D., Jackson, R.W., Seth, A., Kolesar, J.A., Delp S.L. 
% Muscle coordination retraining  inspired by musculoskeletal simulations
% reduces knee contact force. Sci Rep 12, 9842 (2022). 
% https://doi.org/10.1038/s41598-022-13386-9

% Modified by: Anthony Le

%%
function [] = MAIN_StaticOptimizationAPI()
% This main loop allows you to run StaticOptimizationAPI.m

clear all; close all; format compact; clc; fclose all;

% % Path to the data and utility functions. No need to change this, unless
% you rearrange the folder structure, differently from github.
data_dir = '..\data' ; % Base Directory to base results directory.
addpath(genpath('..\src'));

subject_id = 1;
trial_id = 1;

subject_dir = ['\Subject', num2str(subject_id)];
trial_names = {'walking_baseline1', ...                                         % 1
               'walking_FBexp1_finalNoFB1', 'walking_FBexp1_finalFB1', ...      % 2, 3
               'walking_FBcolor1_finalNoFB1', 'walking_FBcolor1_finalFB1', ...  % 4, 5
               'walking_FBcolor2_finalNoFB1', 'walking_FBcolor2_finalFB1', ...  % 6, 7
               'walking_retention1'};                                           % 8
trial_name = trial_names{trial_id};

% % % Fill Path names
INPUTS.trialname = trial_name ; 
INPUTS.forceFilePath = [data_dir subject_dir '\ExpmtlData\GRF\' trial_name '_forces.mot'] ;  % Full path of forces file
INPUTS.ikFilePath = [data_dir subject_dir '\IK\' trial_name '\output\results_ik.sto'] ; % Full path of IK file
INPUTS.idFilePath = [data_dir subject_dir '\ID\' trial_name '\output\results_id.sto'] ; % Full path of ID file
INPUTS.emgFilePath = [data_dir subject_dir '\ProcessedEMG\' trial_name '\EMG_allMuscles.sto'] ; % location of *.mot file with normalized EMG (if using EMG)
INPUTS.outputFilePath = [data_dir subject_dir '\StaticOpt\' trial_name '\'] ; % full path for SO & JRA outputs
INPUTS.modelDir = [data_dir subject_dir '\Models\'] ; % full path to folder where model is
INPUTS.modelName = 'RajagopalModified_scaled.osim' ; % model file name
geometryPath = [] ; % full path to geometry folder for Model. If pointing to Geometry folder in OpenSim install, leave this field blank: []

%%
import org.opensim.modeling.*

temp_storage = Storage([data_dir subject_dir '\ExpmtlData\GRF\' trial_name '_forces.mot']);
time_col = ArrayDouble();

temp_storage.getTimeColumn(time_col);
time = str2num(time_col);
start_time = time(1);
end_time = time(end);

%%
% % % Set time for simulation % % %
INPUTS.startTime = start_time ;
INPUTS.endTime = end_time ;

INPUTS.leg = 'r' ; % If deleteContralateralMuscles flag is true, actuates this leg
                   % with muscles and contralateral leg with coordinate actuators 
                   % only. If deleteContralateralMuscles flag is false,
                   % this input doesn't matter.

% Flags

% % Load up the INPUTS structure for static optimization parameters that are constant across all
% trials and subjects
INPUTS.filtFreq = 6 ; % Lowpass filter frequency for IK coordinates. -1 if no filtering 

% Flags
INPUTS.appendActuators = true ; % Append reserve actuators at all coordinates?
INPUTS.appendForces = true ; % True if you want to append grfs?
INPUTS.deleteContralateralMuscles = false ; % replace muscles on contralateral leg with powerful reserve actuators (makes SO faster)
INPUTS.useEmgRatios = false ; % true if you want to track EMG ratios defined in INPUTS.emgRatioPairs
INPUTS.useEqualMuscles = false ; % true if you want to constrain INPUTS.equalMuscles muscle pairs to be equivalent
INPUTS.useEmgConstraints = false ; % true if you want to constrain muscle activations to follow EMG input INPUTS.emgConstrainedMuscles
INPUTS.changePassiveForce = false ; % true if want to turn passive forces off
INPUTS.ignoreTendonCompliance = false ; % true if making all tendons rigid


% Degrees of Freedom to ignore (patellar coupler constraints, etc.) during moment matching constraint
INPUTS.fixedDOFs = {'knee_angle_r_beta','knee_angle_l_beta'} ;

% EMG file
INPUTS.emgRatioPairs = {} ; % nPairs x 2 cell for muscle names whos ratios you want to constrain with EMG. Can leave off '_[leg]' if you want it to apply to both
INPUTS.equalMuscles = {} ; % nPairs x 2 cell of muscles for whom you want equal activations
INPUTS.emgConstrainedMuscles = {} ; % nMuscles x 1 cell of muscles for which you want activation to track EMG.  Can leave off '_[leg]' if you want it to apply to both

INPUTS.emgSumThreshold = 0 ; % If sum of emg pairs is less than this it won't show up in the constraint or cost (wherever you put it)

% Weights for reserves, muscles. The weight is in
% the cost function as sum(w*(whatever^2)), so the weight is not squared.
INPUTS.reserveActuatorWeights = 1 ; 
INPUTS.muscleWeights = 1 ;
INPUTS.ipsilateralActuatorStrength = 1 ;
INPUTS.contralateralActuatorStrength = 100 ;
INPUTS.weightsToOverride = {} ; % Overrides the general actuator weight for muscles or reserves.
                                % Can be a partial name. Eg. 'hip_rotation' will change hip_rotation_r and hip_rotation_l
                                % or 'gastroc' to override the weight for the right and left gastroc muscles
INPUTS.overrideWeights = [] ; % A column vector the same size as weights  
INPUTS.prescribedActuationCoords = {} ; % A column cell with coordinates (exact name) that will be prescribed from ID moments eg. 'knee_adduction_r' 
                                        % The muscles will not aim to balance the moment at this DOF,
                                        % but their contribution to the moment will be computed at the
                                        % end of the optimization step, and the remaining moment generated by
                                        % the reserve actuator


% External Forces Definitions
INPUTS.externalForceName = {'GRF_r','GRF_l'} ; % nForces x 1 cell
INPUTS.applied_to_body = {'calcn_r','calcn_l'} ; 
INPUTS.force_expressed_in_body =  {'ground','ground'} ;
INPUTS.force_identifier = {'ground_force_v','1_ground_force_v'} ;
INPUTS.point_expressed_in_body = {'ground','ground'} ;
INPUTS.point_identifier = {'ground_force_p','1_ground_force_p'} ;

% Joint Reaction Fields
INPUTS.jRxn.inFrame = 'child' ;
INPUTS.jRxn.onBody = 'child' ;
INPUTS.jRxn.jointNames = ['all'] ;

INPUTS.passiveForceStrains = [3 4] ; % Default = [0,.7] this is strain at zero force and strain at 1 norm force in Millard model
                                     % This only matters if ignorePassiveForces = true

% % % % % END OF USER INPUTS % % % % %% % % % %% % % % %% % % % %% % % % %


if ~isempty(INPUTS.overrideWeights)
    disp('YOU ARE OVERRIDING SOME ACTUATOR WEIGHTS'); 
end

if ~isempty(geometryPath)
    org.opensim.modeling.ModelVisualizer.addDirToGeometrySearchPaths(geometryPath)
end
    
% Run it!
StaticOptimizationAPIVectorized(INPUTS) ; % Run StaticOptimizationAPI

% Save this script in the folder to reference settings
FileNameAndLocation=[mfilename('fullpath')];
newbackup=[INPUTS.outputFilePath 'API_staticOpt_settings.m'];
currentfile=strcat(FileNameAndLocation, '.m');
copyfile(currentfile,newbackup);

end % Main

