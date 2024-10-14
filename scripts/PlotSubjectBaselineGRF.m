%% Plot ground reaction forces for a subject

%% Initialize workspace
clear; close all; clc; fclose all; format compact;

import org.opensim.modeling.*

%%
data_dir = '..\data';
subject_id = '6';
force_file = 'walking_baseline1_forces.mot';
grf_file_path = sprintf('%s\\Subject%s\\ExpmtlData\\GRF\\%s', ...
                        data_dir, subject_id, force_file);

muscle_file = '';

storage = Storage(grf_file_path);

%%
time_col = ArrayDouble();
grf_x_r_col = ArrayDouble();
grf_y_r_col = ArrayDouble();
grf_z_r_col = ArrayDouble();
grf_x_l_col = ArrayDouble();
grf_y_l_col = ArrayDouble();
grf_z_l_col = ArrayDouble();

storage.getTimeColumn(time_col);

storage.getDataColumn('ground_force_vx', grf_x_r_col); 
storage.getDataColumn('ground_force_vy', grf_y_r_col); 
storage.getDataColumn('ground_force_vz', grf_z_r_col);
storage.getDataColumn('1_ground_force_vx', grf_x_l_col); 
storage.getDataColumn('1_ground_force_vy', grf_y_l_col); 
storage.getDataColumn('1_ground_force_vz', grf_z_l_col);

time = str2num(time_col);
grf_x_r = str2num(grf_x_r_col);
grf_y_r = str2num(grf_y_r_col);
grf_z_r = str2num(grf_z_r_col);
grf_x_l = str2num(grf_x_l_col);
grf_y_l = str2num(grf_y_l_col);
grf_z_l = str2num(grf_z_l_col);



%%
figure();
subplot(2, 1, 1);
hold on;
plot(time, grf_x_r, 'LineWidth', 1.5);
plot(time, grf_y_r, 'LineWidth', 1.5);
plot(time, grf_z_r, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Ground Reaction Force (N)');
legend('X_R', 'Y_R', 'Z_R');
hold off;

subplot(2, 1, 2);
hold on;
plot(time, grf_x_l, 'LineWidth', 1.5);
plot(time, grf_y_l, 'LineWidth', 1.5);
plot(time, grf_z_l, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Ground Reaction Force (N)');
legend('X_L', 'Y_L', 'Z_L');
hold off;


