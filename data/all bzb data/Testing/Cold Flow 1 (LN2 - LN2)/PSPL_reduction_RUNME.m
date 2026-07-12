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
[ConvertedData,~,ChanNames,~,~]=convertTDMS(0);
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

for i = chanNum:-1:1 
    chanDataDEV5(i).rawdata = ConvertedData.Data.MeasuredData(i+2).Data;
    chanDataDEV5(i).property = ConvertedData.Data.MeasuredData(i+2).Property;
    
    % Parse Channel Data    
    chanDataDEV5(i).name     = chanDataDEV5(i).property(idxName).Value;   
    chanDataDEV5(i).slope    = chanDataDEV5(i).property(idxSlope).Value;   
    chanDataDEV5(i).offset   = chanDataDEV5(i).property(idxOffset).Value;   
    chanDataDEV5(i).zero     = chanDataDEV5(i).property(idxZero).Value;   
    chanDataDEV5(i).units    = chanDataDEV5(i).property(idxUnit).Value;
    chanDataDEV5(i).desc     = chanDataDEV5(i).property(idxDesc).Value;

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
[ConvertedData,~,ChanNames,~,~]=convertTDMS(0);
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

for i = chanNum:-1:1 
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

%% Assign DEV5 Data

PT_FU_01 = chanDataDEV5(1).rawdata.*chanDataDEV5(1).slope+chanDataDEV5(1).zero+chanDataDEV5(1).offset;
PT_OX_01 = chanDataDEV5(2).rawdata.*chanDataDEV5(2).slope+chanDataDEV5(2).zero+chanDataDEV5(2).offset;
PT_HE_01 = chanDataDEV5(3).rawdata.*chanDataDEV5(3).slope+chanDataDEV5(3).zero+chanDataDEV5(3).offset;
PT_FU_05 = chanDataDEV5(4).rawdata.*chanDataDEV5(4).slope+chanDataDEV5(4).zero+chanDataDEV5(4).offset;
PT_OX_05 = chanDataDEV5(5).rawdata.*chanDataDEV5(5).slope+chanDataDEV5(5).zero+chanDataDEV5(5).offset;
PI_OX_01 = chanDataDEV5(6).rawdata.*chanDataDEV5(6).slope+chanDataDEV5(6).zero+chanDataDEV5(6).offset;
PI_FU_01 = chanDataDEV5(7).rawdata.*chanDataDEV5(7).slope+chanDataDEV5(7).zero+chanDataDEV5(7).offset;
PT_HE_02 = chanDataDEV5(8).rawdata.*chanDataDEV5(8).slope+chanDataDEV5(8).zero+chanDataDEV5(8).offset;
PT_OX_02 = chanDataDEV5(9).rawdata.*chanDataDEV5(9).slope+chanDataDEV5(9).zero+chanDataDEV5(9).offset;
PT_FU_02 = chanDataDEV5(10).rawdata.*chanDataDEV5(10).slope+chanDataDEV5(10).zero+chanDataDEV5(10).offset;
PI_OX_02 = chanDataDEV5(11).rawdata.*chanDataDEV5(11).slope+chanDataDEV5(11).zero+chanDataDEV5(11).offset;
PI_FU_02 = chanDataDEV5(12).rawdata.*chanDataDEV5(12).slope+chanDataDEV5(12).zero+chanDataDEV5(12).offset;
PI_HE_01 = chanDataDEV5(13).rawdata.*chanDataDEV5(13).slope+chanDataDEV5(13).zero+chanDataDEV5(13).offset;
PI_FU_03 = chanDataDEV5(14).rawdata.*chanDataDEV5(14).slope+chanDataDEV5(14).zero+chanDataDEV5(14).offset;
PI_OX_03 = chanDataDEV5(15).rawdata.*chanDataDEV5(15).slope+chanDataDEV5(15).zero+chanDataDEV5(15).offset;
FMS_ENG_01 = chanDataDEV5(16).rawdata.*chanDataDEV5(16).slope+chanDataDEV5(16).zero+chanDataDEV5(16).offset;

% Original Assignment Method by Data Reduction
% TC_FU_01 = chanData(4).rawdata;
% TC_OX_01 = chanData(5).rawdata;

%% Assign DEV6 Data
PT_FU_03 = chanDataDEV6(1).rawdata.*chanDataDEV6(1).slope+chanDataDEV6(1).zero+chanDataDEV6(1).offset;
PT_OX_03 = chanDataDEV6(2).rawdata.*chanDataDEV6(2).slope+chanDataDEV6(2).zero+chanDataDEV6(2).offset;
PT_ENG_01 = chanDataDEV6(3).rawdata.*chanDataDEV6(3).slope+chanDataDEV6(3).zero+chanDataDEV6(3).offset;
PT_N2_01 = chanDataDEV6(4).rawdata.*chanDataDEV6(4).slope+chanDataDEV6(4).zero+chanDataDEV6(4).offset;
PT_OX_04 = chanDataDEV6(5).rawdata.*chanDataDEV6(5).slope+chanDataDEV6(5).zero+chanDataDEV6(5).offset;
PT_FU_04 = chanDataDEV6(6).rawdata.*chanDataDEV6(6).slope+chanDataDEV6(6).zero+chanDataDEV6(6).offset;
% CH7 Skipped
TC_OX_03 = chanDataDEV6(8).rawdata.*chanDataDEV6(8).slope+chanDataDEV6(8).zero+chanDataDEV6(8).offset;
TC_FU_03 = chanDataDEV6(9).rawdata.*chanDataDEV6(9).slope+chanDataDEV6(9).zero+chanDataDEV6(9).offset;
TC_HE_01 = chanDataDEV6(10).rawdata.*chanDataDEV6(10).slope+chanDataDEV6(10).zero+chanDataDEV6(10).offset;
TC_OX_04 = chanDataDEV6(11).rawdata.*chanDataDEV6(11).slope+chanDataDEV6(11).zero+chanDataDEV6(11).offset;
TC_FU_04 = chanDataDEV6(12).rawdata.*chanDataDEV6(12).slope+chanDataDEV6(12).zero+chanDataDEV6(12).offset;
TC_FU_02 = chanDataDEV6(13).rawdata.*chanDataDEV6(13).slope+chanDataDEV6(13).zero+chanDataDEV6(13).offset;
TC_FU_01 = chanDataDEV6(14).rawdata.*chanDataDEV6(14).slope+chanDataDEV6(14).zero+chanDataDEV6(14).offset;
TC_OX_02 = chanDataDEV6(15).rawdata.*chanDataDEV6(15).slope+chanDataDEV6(15).zero+chanDataDEV6(15).offset;
TC_OX_01 = chanDataDEV6(16).rawdata.*chanDataDEV6(16).slope+chanDataDEV6(16).zero+chanDataDEV6(16).offset;

% %% Plot Processed Data

% Braden's Plots
% figure('DefaultAxesFontSize',18); box on; grid on;
% plot(time, PT_FU_01, time, PT_OX_01, time, PT_FU_05, time, PT_OX_05, time, PT_HE_01, time, PT_OX_03,time, PT_FU_03)
% title('Rocket PTs')
% xlabel('Time [s]')
% ylabel('Pressure [psi]')
% legend('PT-FU-01', 'PT-OX-01', 'PT-FU-05', 'PT-OX-05', 'PT-HE-01','PT-FU-03', 'PT-OX-03')
% box on; grid on;
% 
% figure('DefaultAxesFontSize',18); box on; grid on;
% plot(time, PT_HE_02, time, PT_FU_02, time, PT_OX_02, time, PT_N2_01, time, PT_FU_04, time, PT_OX_04 )
% title('Ground PTs')
% xlabel('Time [s]')
% ylabel('Pressure [psi]')
% legend('PT-HE-02', 'PT-FU-02', 'PT-OX-02','PT-N2-01', 'PT-FU-04', 'PT-OX-04')
% box on; grid on;
% 
% figure('DefaultAxesFontSize',18); box on; grid on;
% plot(time, TC_FU_01, time, TC_OX_01, time, TC_HE_01, time, TC_OX_03,time, TC_FU_03)
% title('Rocket TCs')
% xlabel('Time [s]')
% ylabel('Temperature [C]')
% legend('TC-FU-01', 'TC-OX-01','TC-HE-01','TC-FU-03', 'TC-OX-03')
% box on; grid on;
% 
% figure('DefaultAxesFontSize',18); box on; grid on;
% plot(time, TC_FU_02, time, TC_OX_02, time, TC_FU_04, time, TC_OX_04 )
% title('Ground TCs')
% xlabel('Time [s]')
% ylabel('Temperature [C]')
% legend('TC-FU-02', 'TC-OX-02','TC-FU-04', 'TC-OX-04')
% box on; grid on;
