clear
clc

close all

Offset_HE = -5.6;
Offset_FU = -0.8;
Offset_OX = -0.8;

data_kate =  readmatrix('BZB Data_R-Spreadsheet.csv');
Acceleration = data_kate(:,2);

data = readmatrix("Flight_1_PT_Data.txt");

xAxis = [data(:, 1)];
timeStampOverflow = data(:, 2) / 1000000;
timeStamp = zeros(length(timeStampOverflow), 1);
timeStep = zeros(length(timeStampOverflow), 1);

for i = 1 : length(timeStampOverflow) - 1
    if timeStampOverflow(i + 1) - timeStampOverflow(i) > 0
        timeStep(i) = timeStampOverflow(i + 1) - timeStampOverflow(i);
    else
        timeStep(i) = 0.002;
    end
end

for i = 2 : length(timeStep)
    timeStamp(i) = timeStamp(i - 1) + timeStep(i);
end

PT_HE_01 = 6.144 * double(bitshift(int64(data(:, 4)), -4)) / ((2 ^ 11) - 1) * (10 / 3.3) * 501.98;
PT_FU_01 = 6.144 * double(bitshift(int64(data(:, 5)), -4)) / ((2 ^ 11) - 1) * (10 / 3.3) * 100.07;
PT_OX_01 = 6.144 * double(bitshift(int64(data(:, 3)), -4)) / ((2 ^ 11) - 1) * (10 / 3.3) * 100.03;

Acc = [numel(Acceleration) * 5;1];
count = 0;
for i = 1:1:numel(PT_HE_01)
   if (i >= (1494.82 * count * .2))
       count = count + 1;
       Acc(i) = Acceleration(count);
   end
end

CPU_Temp = data(:, 6);
Autosequence = data(:, 7);



%offsets
PT_HE_01 = PT_HE_01 - Offset_HE;
PT_FU_01 = PT_FU_01 - Offset_FU;
PT_OX_01 = PT_OX_01 - Offset_OX;


%12860 to 12920
Figure_ALT_T = figure('Name','Tank pressure vs Time');
Plot_Alt_T = plot(timeStamp(),PT_HE_01(),timeStamp(),PT_FU_01(),timeStamp(),PT_OX_01());
title('Tank Pressure vs Time');
xlabel('Time (seconds)');
ylabel('Tank pressure (psia)');
xlim([1420 1560]);
legend('PT-HE-01','PT-FU-01','PT-OX-01');
grapherPSP(Figure_ALT_T,Plot_Alt_T,'Light');

Figure_ALT_T = figure('Name','Tank pressure vs Time');
Plot_Alt_T = plot(timeStamp(),PT_HE_01(),timeStamp(),PT_FU_01(),timeStamp(),PT_OX_01(),timeStamp(),Acc());
title('Tank Pressure vs Time');
xlabel('Time (seconds)');
ylabel('Tank pressure (psia)');
xlim([1480 1560]);
legend('PT-HE-01','PT-FU-01','PT-OX-01');
grapherPSP(Figure_ALT_T,Plot_Alt_T,'Light');

% plots only prop
Figure_ALT_T = figure('Name','Tank Pressure vs Time');
Plot_Alt_T = plot(timeStamp(),PT_FU_01(),timeStamp(),PT_OX_01());
title('Tank Pressure vs Time');
xlabel('Time (seconds)');
ylabel('Tank pressure (psia)');
xlim([1480 1560]);
legend('PT-FU-01','PT-OX-01');
grapherPSP(Figure_ALT_T,Plot_Alt_T,'Light');