%  PSP-L DATA REDUCTION CODE
%  For: C-DAQ 
%  10/28/2021
%  Adapted from existing lab codes
 

 

% %% Initialization
% clear all; close all; clc;
% %% Select, convert, and pull data DEV1
% [ConvertedData,ConvertVer,ChanNames,GroupNames,ci]=convertTDMS(0);
% TestDate = extractBetween(ConvertedData.FileName,9,17);
% TestName = extractBetween(ConvertedData.FileName,27,length(ConvertedData.FileName)-5);
% chanNum = max(size(ChanNames{1}));
% tempProp = extractfield(ConvertedData.Data.MeasuredData(3).Property, 'Name');
% idxName  = find( strcmp(tempProp, 'Channel Name'));
% idxSlope = find( strcmp(tempProp, 'Slope'));
% idxOffset= find( strcmp(tempProp, 'Offset'));
% idxZero  = find( strcmp(tempProp, 'Zeroing Correction'));
% idxUnit = find( strcmp(tempProp, 'Unit'));
% idxDesc = find( strcmp(tempProp, 'Description'));
% for i = 1:chanNum 
%     chanData(i).rawdata = ConvertedData.Data.MeasuredData(i+2).Data;
%     chanData(i).property = ConvertedData.Data.MeasuredData(i+2).Property;
%     
%     % Parse Channel Data    
%     chanData(i).name     = chanData(i).property(idxName).Value;   
%     chanData(i).slope    = chanData(i).property(idxSlope).Value;   
%     chanData(i).offset   = chanData(i).property(idxOffset).Value;   
%     chanData(i).zero     = chanData(i).property(idxZero).Value;   
%     chanData(i).units    = chanData(i).property(idxUnit).Value;
%     chanData(i).desc     = chanData(i).property(idxDesc).Value;
% 
% end
% nameSplit = strsplit(ChanNames{1}{1}, {'(',')'});
% scan_rate = str2num(cell2mat(strsplit(nameSplit{2}, ' Hz')));
% samps = ConvertedData.Data.MeasuredData(3).Total_Samples;
% dt = 1/scan_rate;
% time  = 0:dt:(samps-1)*dt;
% 
% % Trash Collection
% clear ConvertedData ConvertVer ChanNames GroupNames ci idxName ...
%     idxSlope idxOffset idxZero idxUnit idxDesc






%% Get data variables
PT_FU_07 = chanData(1).rawdata.*chanData(1).slope+chanData(1).zero+chanData(1).offset;
PT_FU_06 = chanData(2).rawdata.*chanData(2).slope+chanData(2).zero+chanData(2).offset;
TC_LN2_01 = chanData(3).rawdata.*chanData(3).slope+chanData(3).zero+chanData(3).offset;
TC_LN2_02 = chanData(4).rawdata.*chanData(4).slope+chanData(4).zero+chanData(4).offset;
TC_LN2_03 = chanData(5).rawdata.*chanData(5).slope+chanData(5).zero+chanData(5).offset;
TC_LN2_04 = chanData(6).rawdata.*chanData(6).slope+chanData(6).zero+chanData(6).offset;
TC_FU_06 = chanData(7).rawdata.*chanData(7).slope+chanData(7).zero+chanData(7).offset;
PI_FU_04 = chanData(8).rawdata.*chanData(8).slope+chanData(8).zero+chanData(8).offset;
PI_FU_05 = chanData(9).rawdata.*chanData(9).slope+chanData(9).zero+chanData(9).offset;
PI_LN2_01 = chanData(10).rawdata.*chanData(10).slope+chanData(10).zero+chanData(10).offset;
PI_LN2_02 = chanData(11).rawdata.*chanData(11).slope+chanData(11).zero+chanData(11).offset;
TC_FU_04 = chanData(12).rawdata.*chanData(12).slope+chanData(12).zero+chanData(12).offset;
PT_FU_04 = chanData(13).rawdata.*chanData(13).slope+chanData(13).zero+chanData(13).offset;
NV_LN2_01 = chanData(14).rawdata.*chanData(14).slope+chanData(14).zero+chanData(14).offset;
PT_LN2_01 = chanData(15).rawdata.*chanData(15).slope+chanData(15).zero+chanData(15).offset;
% N/A = chanData(16).rawdata.*chanData(16).slope+chanData(16).zero+chanData(16).offset;


%% Plots
figure(1); box on; grid on;
plot(time, PT_FU_07, time, PT_FU_06)
title('FUEL PTs')
xlabel('Time [s]')
ylabel('Pressure [psi]')
legend('PT-FU-07', 'PT-FU-06')
box on; grid on;

figure(2); box on; grid on;
plot(time, PT_LN2_01)
title('LN2 PT')
xlabel('Time [s]')
ylabel('Pressure [psi]')
legend('PT-LN2-01')
box on; grid on;

figure(3); box on; grid on;
plot(time, TC_LN2_01, time, TC_LN2_02, time, TC_LN2_03, time, TC_LN2_04)
title('LN2 JACKET TCs')
xlabel('Time [s]')
ylabel('Temperature [F]')
legend('TC-LN2-01', 'TC-LN2-02','TC-LN2-03','TC-LN2-04')
box on; grid on;

figure(4); box on; grid on;
plot(time, TC_FU_06)
title('FUEL TC')
xlabel('Time [s]')
ylabel('Temperature [F]')
legend('TC-FU-06')
box on; grid on;


