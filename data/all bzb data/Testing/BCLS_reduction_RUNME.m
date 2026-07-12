%  PSP-L DATA REDUCTION CODE
%  For: BCLS 
%  10/07/2021
%  Adapted from existing lab codes
%  
%
%  ***Select the high frequency file second.***
%
%% Initialization
clear all; close all; clc;
%% Select, convert, and pull data DEV5
[ConvertedData,ConvertVer,ChanNames,GroupNames,ci]=convertTDMS(0);
TestDate = extractBetween(ConvertedData.FileName,9,17);
TestName = extractBetween(ConvertedData.FileName,27,length(ConvertedData.FileName)-5);
chanNum = max(size(ChanNames{1}));
tempProp = extractfield(ConvertedData.Data.MeasuredData(3).Property, 'Name');
idxName  = find( strcmp(tempProp, 'Channel Name'));
idxSlope = find( strcmp(tempProp, 'Slope'));
idxOffset= find( strcmp(tempProp, 'Offset'));
idxZero  = find( strcmp(tempProp, 'Zeroing Correction'));
idxUnit = find( strcmp(tempProp, 'Unit'));
idxDesc = find( strcmp(tempProp, 'Description'));
for i = 1:chanNum 
    chanData(i).rawdata = ConvertedData.Data.MeasuredData(i+2).Data;
    chanData(i).property = ConvertedData.Data.MeasuredData(i+2).Property;
    
    % Parse Channel Data    
    chanData(i).name     = chanData(i).property(idxName).Value;   
    chanData(i).slope    = chanData(i).property(idxSlope).Value;   
    chanData(i).offset   = chanData(i).property(idxOffset).Value;   
    chanData(i).zero     = chanData(i).property(idxZero).Value;   
    chanData(i).units    = chanData(i).property(idxUnit).Value;
    chanData(i).desc     = chanData(i).property(idxDesc).Value;

end
nameSplit = strsplit(ChanNames{1}{1}, {'(',')'});
scan_rate = str2num(cell2mat(strsplit(nameSplit{2}, ' Hz')));
samps = ConvertedData.Data.MeasuredData(3).Total_Samples;
dt = 1/scan_rate;
time  = 0:dt:(samps-1)*dt;

% Trash Collection
clear ConvertedData ConvertVer ChanNames GroupNames ci idxName ...
    idxSlope idxOffset idxZero idxUnit idxDesc

%% Select, convert, and pull DEV 6
[ConvertedData,ConvertVer,ChanNames,GroupNames,ci]=convertTDMS(0);
TestDate = extractBetween(ConvertedData.FileName,9,17);
TestName = extractBetween(ConvertedData.FileName,27,length(ConvertedData.FileName)-5);
chanNum = max(size(ChanNames{1}));
tempProp = extractfield(ConvertedData.Data.MeasuredData(3).Property, 'Name');
idxName  = find( strcmp(tempProp, 'Channel Name'));
idxSlope = find( strcmp(tempProp, 'Slope'));
idxOffset= find( strcmp(tempProp, 'Offset'));
idxZero  = find( strcmp(tempProp, 'Zeroing Correction'));
idxUnit = find( strcmp(tempProp, 'Unit'));
idxDesc = find( strcmp(tempProp, 'Description'));
for i = 1:chanNum 
    chanDataDEV6(i).rawdata = ConvertedData.Data.MeasuredData(i+2).Data;
    chanDataDEV6(i).property = ConvertedData.Data.MeasuredData(i+2).Property;
    
    % Parse Channel Data    
    chanDataDEV6(i).name     = chanDataDEV6(i).property(idxName).Value;   
    chanDataDEV6(i).slope    = chanDataDEV6(i).property(idxSlope).Value;   
    chanDataDEV6(i).offset   = chanDataDEV6(i).property(idxOffset).Value;   
    chanDataDEV6(i).zero     = chanDataDEV6(i).property(idxZero).Value;   
    chanDataDEV6(i).units    = chanDataDEV6(i).property(idxUnit).Value;
    chanDataDEV6(i).desc     = chanDataDEV6(i).property(idxDesc).Value;

end
nameSplit = strsplit(ChanNames{1}{1}, {'(',')'});
scan_rate = str2num(cell2mat(strsplit(nameSplit{2}, ' Hz')));
samps = ConvertedData.Data.MeasuredData(3).Total_Samples;
dt = 1/scan_rate;
timeDEV6  = 0:dt:(samps-1)*dt;

% Trash Collection
clear ConvertedData ConvertVer ChanNames GroupNames ci idxName ...
    idxSlope idxOffset idxZero idxUnit idxDesc

%% Plot DEV5 Data

PT_FU_01 = chanData(1).rawdata.*chanData(1).slope+chanData(1).zero+chanData(1).offset;
PT_OX_01 = chanData(2).rawdata.*chanData(2).slope+chanData(2).zero+chanData(2).offset;;
PT_HE_01 = chanData(3).rawdata.*chanData(3).slope+chanData(3).zero+chanData(3).offset;;
PT_FU_05 = chanData(4).rawdata.*chanData(4).slope+chanData(4).zero+chanData(4).offset;;
PT_OX_05 = chanData(5).rawdata.*chanData(5).slope+chanData(5).zero+chanData(5).offset;;
PT_HE_02 = chanData(8).rawdata.*chanData(8).slope+chanData(8).zero+chanData(8).offset;;
PT_OX_02 = chanData(9).rawdata.*chanData(9).slope+chanData(9).zero+chanData(9).offset;;
PT_FU_02 = chanData(10).rawdata.*chanData(10).slope+chanData(10).zero+chanData(10).offset;;
FMS_ENG_01 = chanData(11).rawdata.*chanData(11).slope+chanData(11).zero+chanData(11).offset;;

% TC_FU_01 = chanData(4).rawdata;
% TC_OX_01 = chanData(5).rawdata;




%% Plot DEV6 Data
PT_FU_03 = chanDataDEV6(1).rawdata.*chanDataDEV6(1).slope+chanDataDEV6(1).zero+chanDataDEV6(1).offset;
PT_OX_03 = chanDataDEV6(2).rawdata.*chanDataDEV6(2).slope+chanDataDEV6(2).zero+chanDataDEV6(2).offset;
PT_ENG_01 = chanDataDEV6(3).rawdata.*chanDataDEV6(3).slope+chanDataDEV6(3).zero+chanDataDEV6(3).offset;
PT_N2_01 = chanDataDEV6(4).rawdata.*chanDataDEV6(4).slope+chanDataDEV6(4).zero+chanDataDEV6(4).offset;
PT_OX_04 = chanDataDEV6(5).rawdata.*chanDataDEV6(5).slope+chanDataDEV6(5).zero+chanDataDEV6(5).offset;
PT_FU_04 = chanDataDEV6(6).rawdata.*chanDataDEV6(6).slope+chanDataDEV6(6).zero+chanDataDEV6(6).offset;
% TC_OX_03 = chanDataDEV6(7).rawdata.*chanDataDEV6(7).slope+chanDataDEV6(7).zero+chanDataDEV6(7).offset;
TC_OX_03 = chanDataDEV6(8).rawdata.*chanDataDEV6(8).slope+chanDataDEV6(8).zero+chanDataDEV6(8).offset;
TC_FU_03 = chanDataDEV6(9).rawdata.*chanDataDEV6(9).slope+chanDataDEV6(9).zero+chanDataDEV6(9).offset;
TC_HE_01 = chanDataDEV6(10).rawdata.*chanDataDEV6(10).slope+chanDataDEV6(10).zero+chanDataDEV6(10).offset;
TC_OX_04 = chanDataDEV6(11).rawdata.*chanDataDEV6(11).slope+chanDataDEV6(11).zero+chanDataDEV6(11).offset;
TC_FU_04 = chanDataDEV6(12).rawdata.*chanDataDEV6(12).slope+chanDataDEV6(12).zero+chanDataDEV6(12).offset;
TC_FU_02 = chanDataDEV6(13).rawdata.*chanDataDEV6(13).slope+chanDataDEV6(13).zero+chanDataDEV6(13).offset;
TC_FU_01 = chanDataDEV6(14).rawdata.*chanDataDEV6(14).slope+chanDataDEV6(14).zero+chanDataDEV6(14).offset;
TC_OX_02 = chanDataDEV6(15).rawdata.*chanDataDEV6(15).slope+chanDataDEV6(15).zero+chanDataDEV6(15).offset;
TC_OX_01 = chanDataDEV6(16).rawdata.*chanDataDEV6(16).slope+chanDataDEV6(16).zero+chanDataDEV6(16).offset;

figure('DefaultAxesFontSize',18); box on; grid on;
plot(time, PT_FU_01, time, PT_OX_01, time, PT_FU_05, time, PT_OX_05, time, PT_HE_01, time, PT_OX_03,time, PT_FU_03)
title('Rocket PTs')
xlabel('Time [s]')
ylabel('Pressure [psi]')
legend('PT-FU-01', 'PT-OX-01', 'PT-FU-05', 'PT-OX-05', 'PT-HE-01','PT-FU-03', 'PT-OX-03')
box on; grid on;

figure('DefaultAxesFontSize',18); box on; grid on;
plot(time, PT_HE_02, time, PT_FU_02, time, PT_OX_02, time, PT_N2_01, time, PT_FU_04, time, PT_OX_04 )
title('Ground PTs')
xlabel('Time [s]')
ylabel('Pressure [psi]')
legend('PT-HE-02', 'PT-FU-02', 'PT-OX-02','PT-N2-01', 'PT-FU-04', 'PT-OX-04')
box on; grid on;

figure('DefaultAxesFontSize',18); box on; grid on;
plot(time, TC_FU_01, time, TC_OX_01, time, TC_HE_01, time, TC_OX_03,time, TC_FU_03)
title('Rocket TCs')
xlabel('Time [s]')
ylabel('Temperature [C]')
legend('TC-FU-01', 'TC-OX-01','TC-HE-01','TC-FU-03', 'TC-OX-03')
box on; grid on;

figure('DefaultAxesFontSize',18); box on; grid on;
plot(time, TC_FU_02, time, TC_OX_02, time, TC_FU_04, time, TC_OX_04 )
title('Ground TCs')
xlabel('Time [s]')
ylabel('Temperature [C]')
legend('TC-FU-02', 'TC-OX-02','TC-FU-04', 'TC-OX-04')
box on; grid on;

% %% Plot Processed Data
