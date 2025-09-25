%% Plot ground reaction forces for a subject

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
trans1 = find(diff(grf_y_r > 0) == 1);
trans2 = find(diff(grf_y_r ~= 0) == -1);


%%
figure();
hold on;
plot(time_grf, grf_y_r, 'LineWidth', 1.5);
plot(time_muscle, tibpost_r, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Force (N)');
legend({'grf_y_r', 'tibpost_r'}, 'Interpreter', 'None');
hold off;


%%
figure();
subplot(2, 1, 1);
hold on;
plot(time_grf, grf_x_r, 'LineWidth', 1.5);
plot(time_grf, grf_y_r, 'LineWidth', 1.5);
plot(time_grf, grf_z_r, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Ground Reaction Force (N)');
legend('X_R', 'Y_R', 'Z_R');
hold off;

subplot(2, 1, 2);
hold on;
plot(time_grf, grf_x_l, 'LineWidth', 1.5);
plot(time_grf, grf_y_l, 'LineWidth', 1.5);
plot(time_grf, grf_z_l, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Ground Reaction Force (N)');
legend('X_L', 'Y_L', 'Z_L');
hold off;


