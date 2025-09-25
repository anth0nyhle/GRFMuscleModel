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
tibant_r_col = ArrayDouble();

edl_r_col = ArrayDouble();
ehl_r_col = ArrayDouble();

fdl_r_col = ArrayDouble();
fhl_r_col = ArrayDouble();

gaslat_r_col = ArrayDouble();
gasmed_r_col = ArrayDouble();
soleus_r_col = ArrayDouble();

perbrev_r_col = ArrayDouble();
perlong_r_col = ArrayDouble();

muscle_storage.getTimeColumn(time_muscle_col);
muscle_storage.getDataColumn('tibpost_r', tibpost_r_col);
muscle_storage.getDataColumn('tibant_r', tibant_r_col);
muscle_storage.getDataColumn('edl_r', edl_r_col);
muscle_storage.getDataColumn('ehl_r', ehl_r_col);
muscle_storage.getDataColumn('fdl_r', fdl_r_col);
muscle_storage.getDataColumn('fhl_r', fhl_r_col);
muscle_storage.getDataColumn('gaslat_r', gaslat_r_col);
muscle_storage.getDataColumn('gasmed_r', gasmed_r_col);
muscle_storage.getDataColumn('soleus_r', soleus_r_col);
muscle_storage.getDataColumn('perbrev_r', perbrev_r_col);
muscle_storage.getDataColumn('perlong_r', perlong_r_col);

time_muscle = str2num(time_muscle_col);
tibpost_r = str2num(tibpost_r_col);
tibant_r = str2num(tibant_r_col);
edl_r = str2num(edl_r_col);
ehl_r = str2num(ehl_r_col);
fdl_r = str2num(fdl_r_col);
fhl_r = str2num(fhl_r_col);
gaslat_r = str2num(gaslat_r_col);
gasmed_r = str2num(gasmed_r_col);
soleus_r = str2num(soleus_r_col);
perbrev_r = str2num(perbrev_r_col);
perlong_r = str2num(perlong_r_col);

achilles_r = sum([gaslat_r; gasmed_r; soleus_r], 1);

%%
start_peak = find(diff(grf_y_r > 0) == 1);
end_peak = find(diff(grf_y_r ~= 0) == -1);

num_segments = min(length(start_peak), length(end_peak));

grf_segments = cell(num_segments, 1);
tibpost_r_segments = cell(num_segments, 1);
tibant_r_segments = cell(num_segments, 1);
edl_r_segments = cell(num_segments, 1);
ehl_r_segments = cell(num_segments, 1);
fdl_r_segments = cell(num_segments, 1);
fhl_r_segments = cell(num_segments, 1);
achilles_r_segments = cell(num_segments, 1);
perbrev_r_segments = cell(num_segments, 1);
perlong_r_segments = cell(num_segments, 1);

%%
figure();
hold on;
for i = 1:num_segments
    grf_segments{i} = grf_y_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, grf_y_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Vertical Ground Reaction Force (N)')
hold off;

%%
figure();
subplot(3, 3, 1);
hold on;
for i = 1:num_segments
    tibpost_r_segments{i} = tibpost_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, tibpost_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Tibialis Posterior Force (N)')
hold off;

subplot(3, 3, 2);
hold on;
for i = 1:num_segments
    tibant_r_segments{i} = tibant_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, tibant_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Tibialis Anterior Force (N)')
hold off;

subplot(3, 3, 8);
hold on;
for i = 1:num_segments
    edl_r_segments{i} = edl_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, edl_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Extersor Digitorum Longus Force (N)')
hold off;

subplot(3, 3, 5);
hold on;
for i = 1:num_segments
    ehl_r_segments{i} = ehl_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, ehl_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Extersor Hallucis Longus Force (N)')
hold off;

subplot(3, 3, 7);
hold on;
for i = 1:num_segments
    fdl_r_segments{i} = fdl_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, fdl_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Flexor Digitorum Longus Force (N)')
hold off;

subplot(3, 3, 4);
hold on;
for i = 1:num_segments
    fhl_r_segments{i} = fhl_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, fhl_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Flexor Hallucis Longus Force (N)')
hold off;

subplot(3, 3, 3);
hold on;
for i = 1:num_segments
    achilles_r_segments{i} = achilles_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, achilles_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Achilles Force (N)')
hold off;

subplot(3, 3, 6);
hold on;
for i = 1:num_segments
    perbrev_r_segments{i} = perbrev_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, perbrev_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Peroneus Brevis Force (N)')
hold off;

subplot(3, 3, 9);
hold on;
for i = 1:num_segments
    perlong_r_segments{i} = perlong_r(start_peak(i)-5:end_peak(i)+5);

    x = 1:length(start_peak(i)-5:end_peak(i)+5);

    plot(x, perlong_r(start_peak(i)-5:end_peak(i)+5));
end
xlabel('Time (s)');
ylabel('Peroneus Longus Force (N)')
hold off;
