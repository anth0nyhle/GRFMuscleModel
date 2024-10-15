%% Segment ground reaction force for a subject

%% Initialize workspace
clear; close all; clc; fclose all; format compact;

import org.opensim.modeling.*

%%
data_dir = '..\data';
subject_id = 1;
trial_name = 'walking_baseline1';


grf_file = 'walking_baseline1_forces.mot';
grf_file_path = sprintf('%s\\Subject%s\\ExpmtlData\\GRF\\%s', ...
                        data_dir, num2str(subject_id), grf_file);
grf_storage = Storage(grf_file_path);



muscle_file = 'results_forces.sto';
muscle_force_file_path = sprintf('%s\\Subject%s\\StaticOpt\\%s\\%s', ...
                                 data_dir, num2str(subject_id), trial_name, ...
                                 muscle_file);
muscle_storage = Storage(muscle_force_file_path);


%%
time_grf_col = ArrayDouble();
grf_x_r_col = ArrayDouble();
grf_y_r_col = ArrayDouble();
grf_z_r_col = ArrayDouble();
grf_x_l_col = ArrayDouble();
grf_y_l_col = ArrayDouble();
grf_z_l_col = ArrayDouble();

grf_storage.getTimeColumn(time_grf_col);

grf_storage.getDataColumn('ground_force_vx', grf_x_r_col); 
grf_storage.getDataColumn('ground_force_vy', grf_y_r_col); 
grf_storage.getDataColumn('ground_force_vz', grf_z_r_col);
grf_storage.getDataColumn('1_ground_force_vx', grf_x_l_col); 
grf_storage.getDataColumn('1_ground_force_vy', grf_y_l_col); 
grf_storage.getDataColumn('1_ground_force_vz', grf_z_l_col);

time_grf = str2num(time_grf_col);
grf_x_r = str2num(grf_x_r_col);
grf_y_r = str2num(grf_y_r_col);
grf_z_r = str2num(grf_z_r_col);
grf_x_l = str2num(grf_x_l_col);
grf_y_l = str2num(grf_y_l_col);
grf_z_l = str2num(grf_z_l_col);

%%
time_muscle_col = ArrayDouble();
tibpost_r_col = ArrayDouble();

muscle_storage.getTimeColumn(time_muscle_col);
muscle_storage.getDataColumn('tibpost_r', tibpost_r_col);

time_muscle = str2num(time_muscle_col);
tibpost_r = str2num(tibpost_r_col);

%%
start_peak = find(diff(grf_y_r > 0) == 1);
end_peak = find(diff(grf_y_r ~= 0) == -1);

num_segments = min(length(start_peak), length(end_peak));

segments = cell(num_segments, 1);

figure();
hold on;

for i = 1:num_segments
    segments{i} = grf_y_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, grf_y_r(start_peak(i)-5:end_peak(i)+5));
end
hold off;

