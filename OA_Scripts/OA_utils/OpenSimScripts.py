import os
import opensim as osim
from .OAPreprocessingScripts import filter_ik, filter_id
import gc

def scale_generic(root_dir: str, mass: float, static_pose_filename: str):
    dir = root_dir
    os.chdir(dir)
    subject_id = static_pose_filename.split('/transformed/')[1].split('_walk_static')[0]
    #scale generic model
    setup = 'generic_scale_setup.xml'
    scale_tool = osim.ScaleTool(setup)
    scale_tool.setSubjectMass(mass)
    scale_tool.setName(f'{subject_id}_scaled')
    #set the path to the generic model
    model_maker = scale_tool.getGenericModelMaker()
    model_maker.setModelFileName('Models/RajagopalModified_generic.osim')
    #model_maker.setMarkerSetFileName()
    #set marker file for model scaler
    model_scaler = scale_tool.getModelScaler()
    model_scaler.setMarkerFileName(static_pose_filename)
    #access marker placer object and set inputs
    marker_placer = scale_tool.getMarkerPlacer()
    marker_placer.setStaticPoseFileName(static_pose_filename)
    marker_placer.setOutputModelFileName(dir + f'/Results/Scaling/{subject_id}_scaled.osim')
    scale_tool.run()
    del scale_tool
    gc.collect()

def inverse_kinmatics(root_dir: str, tracking_data_filepath: str, model: osim.Model):
    dir = root_dir
    os.chdir(dir) 
    after_trans = tracking_data_filepath.split('/transformed/')[1]
    subj_trial_speed = after_trans.split('_transformed')[0]
    #subj = subj_trial_speed.split('_')[0]
    #run inverse kinematics
    setup = 'generic_ik_setup.xml'
    ik_tool = osim.InverseKinematicsTool(setup)
    ik_tool.set_report_marker_locations(False)
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(tracking_data_filepath)
    ik_tool.setOutputMotionFileName(f'Results/IK/raw/{subj_trial_speed}_ik.mot')
    ik_tool.run()
    #filter IK results
    filter_ik(dir+ f'/Results/IK/raw/{subj_trial_speed}_ik.mot', dir + f'/Results/IK/filtered/{subj_trial_speed}_ik_filtered.mot')
    del ik_tool
    gc.collect()

def inverse_dynamics(root_dir: str, force_data_filepath: str, tracking_data_filepath:str, model: osim.Model):
    dir = root_dir
    os.chdir(dir)
    after_trans =  tracking_data_filepath.split('/transformed/')[1]
    subj_trial_speed = after_trans.split('_transformed')[0]
    sub = subj_trial_speed.split('_')[0]
    #plug proper grf data into external loads file
    loads = osim.ExternalLoads('generic_externalLoads.xml', True)
    loads.setDataFileName(force_data_filepath)
    loads_path = os.path.join(dir, 'loads', f'{subj_trial_speed}_externalLoads.xml')
    loads.printToXML(loads_path)
    #run inverse dynamics
    id_tool = osim.InverseDynamicsTool('generic_id_setup.xml')
    id_tool.setModel(model)
    id_tool.setExternalLoadsFileName(loads_path)
    ik_file = os.path.join(dir, f'Results/IK/filtered/{subj_trial_speed}_ik_filtered.mot')
    id_tool.setCoordinatesFileName(ik_file)
    id_tool.set_results_directory(dir + '/Results/ID/raw/')
    id_tool.setOutputGenForceFileName(f'{subj_trial_speed}_id.mot')
    # id_tool.setStartTime(start_time)
    # id_tool.setEndTime(end_time)
    id_tool.run()
    filter_id(dir+f'Results/ID/raw/{subj_trial_speed}_id.mot', dir + f'/Results/ID/filtered/{subj_trial_speed}_id_filtered.mot')
    del id_tool
    gc.collect()
