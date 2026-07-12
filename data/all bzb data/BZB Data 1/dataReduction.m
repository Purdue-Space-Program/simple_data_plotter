%% Data Import
c = ConversionDefinitions();
load("nominal6DOF.mat");
addpath "Flight 1" "Flight 2"
kateData_1.raw = readmatrix ("Flight 1\BZB Data_R-Spreadsheet.csv");
kateData_2.raw = readmatrix("Flight 2\Flight 2 Data_R-Spreadsheet.csv");

ravenData_1_1.raw = readmatrix ("Flight 1\Raven 1 FL 1 compiled.csv");
ravenData_2_1.raw = readmatrix ("Flight 2\Raven flight data\Raven 1 FL 2 compiled.csv");
ravenData_2_2.raw = readmatrix ("Flight 2\Raven flight data\Raven 2 FL 2 compiled.csv");

%% Flight 1
kateData_1.time = kateData_1.raw(:, 1); % flight time [s]
kateData_1.acc = kateData_1.raw(:, 2); % acceleration [g's]
kateData_1.vel = kateData_1.raw(:, 3); % velo accelerometer [ft/s]
kateData_1.altAccelo = kateData_1.raw(:, 4); % accelerometer alt [AGL ft]
kateData_1.altGPS = kateData_1.raw(:, 5); % gps alt [AGL ft]
kateData_1.velo3D = kateData_1.raw(:, 7); % gps 3d velo [ft/s]
kateData_1.mach = kateData_1.raw(:, 8); % gps 3d mach
kateData_1.grndSpd = kateData_1.raw(:, 10); % gps ground speed [ft/s]
kateData_1.vertSpd = kateData_1.raw(:, 9); % gps vertical speed [ft/s]
kateData_1.heading = kateData_1.raw(:, 11); % heading [deg]
kateData_1.tilt = kateData_1.raw(:, 12); % title [deg]
kateData_1.latitude = kateData_1.raw(:, 13); % lat [deg]
kateData_1.longitude = kateData_1.raw(:, 14); % long [deg]

ravenData_1_1.accAxial = ravenData_1_1.raw(:, 1:2); % axial acceleration [g's]
ravenData_1_1.accRadial = ravenData_1_1.raw(:, 10:11); % radial acceleration [g's]
ravenData_1_1.alt = ravenData_1_1.raw(:, 34:35); % altitude from accelerometer [ft]
ravenData_1_1.altBaro = ravenData_1_1.raw(:, 37:38); % altitude from barometer [AGL ft]
ravenData_1_1.velo1 = ravenData_1_1.raw(:, 16:17); % velocity 1 [ft/s] 
ravenData_1_1.velo2 = ravenData_1_1.raw(:, 43:44); % velocity 2 [ft/s] <-- this stops acquiring data at 1.3 sec

%% Flight 2
kateData_2.time = kateData_2.raw(:, 1); % flight time [s]
kateData_2.acc = kateData_2.raw(:, 2); % acceleration [g's]
kateData_2.vel = kateData_2.raw(:, 3); % velo accelerometer [ft/s]
kateData_2.altAccelo = kateData_2.raw(:, 4); % accelerometer alt [AGL ft]
kateData_2.altGPS = kateData_2.raw(:, 5); % gps alt [AGL ft]
kateData_2.velo3D = kateData_2.raw(:, 7); % gps 3d velo [ft/s]
kateData_2.mach = kateData_2.raw(:, 8); % gps 3d mach
kateData_2.grndSpd = kateData_2.raw(:, 10); % gps ground speed [ft/s]
kateData_2.vertSpd = kateData_2.raw(:, 9); % gps vertical speed [ft/s]
kateData_2.heading = kateData_2.raw(:, 11); % heading [deg]
kateData_2.tilt = kateData_2.raw(:, 12); % title [deg]
kateData_2.latitude = kateData_2.raw(:, 13); % lat [deg]
kateData_2.longitude = kateData_2.raw(:, 14); % long [deg]

ravenData_2_1.accAxial = ravenData_2_1.raw(:, 1:2); % axial acceleration [g's]
ravenData_2_1.accRadial = ravenData_2_1.raw(:, 10:11); % radial acceleration [g's]
ravenData_2_1.alt = ravenData_2_1.raw(:, 34:35); % altitude from accelerometer [ft]
ravenData_2_1.altBaro = ravenData_2_1.raw(:, 37:38); % altitude from barometer [AGL ft]
ravenData_2_1.velo1 = ravenData_2_1.raw(:, 16:17); % velocity 1 [ft/s] 
ravenData_2_1.velo2 = ravenData_2_1.raw(:, 43:44); % velocity 2 [ft/s]

ravenData_2_2.accAxial = ravenData_2_2.raw(:, 1:2); % axial acceleration [g's]
ravenData_2_2.accRadial = ravenData_2_2.raw(:, 10:11); % radial acceleration [g's]
ravenData_2_2.alt = ravenData_2_2.raw(:, 34:35); % altitude from accelerometer [ft]
ravenData_2_2.altBaro = ravenData_2_2.raw(:, 37:38); % altitude from barometer [AGL ft]
ravenData_2_2.velo1 = ravenData_2_2.raw(:, 16:17); % velocity 1 [ft/s] 
ravenData_2_2.velo2 = ravenData_2_2.raw(:, 43:44); % velocity 2 [ft/s]

%% Plots
f = figure();
p = plot(position_inertial.time(:), position_inertial.signals.values(:, 1) - 2000, 'LineWidth', 2);
hold on;
plot(kateData_1.time(1:194), kateData_1.altGPS(1:194), 'LineWidth', 2);
plot(kateData_2.time(1:160), kateData_2.altGPS(1:160), 'LineWidth', 2);
grid on;
ylabel ("Altitude AGL (feet)");
xlabel ("Time (seconds)");
xlim([0, 50]);
ylim([0, 35e3]);
title ("Nominal Flight Comparison - Altitude");
legend (["Nominal Flight", "Flight 1", "Flight 2"], 'Location','best');
PSPStyler(f, p, 'Light');

f = figure();
p = plot(velocity_inertial.time - 0.023, velocity_inertial.signals.values(:, 1), 'LineWidth', 2);
hold on;
grid on;
plot(kateData_1.time(1:194), kateData_1.vel(1:194), 'LineWidth', 2);
plot(kateData_2.time(1:160), kateData_2.vel(1:160), 'LineWidth', 2);
grid on;
ylabel ("Velocity (feet/s)");
xlabel ("Time (seconds)");
xlim([0, 50]);
ylim([0, 1700]);
title ("Nominal Flight Comparison - Velocity");
legend (["Nominal Flight", "Flight 1", "Flight 2"], 'Location','best');
PSPStyler(f, p, 'Light');

f = figure();
p = plot(ravenData_1_1.accRadial(:, 1) ./ ravenData_1_1.accRadial(534, 1), ravenData_1_1.accRadial(:, 2));
hold on;
plot(ravenData_2_2.accRadial(:, 1) ./ ravenData_2_2.accRadial(70740, 1), ravenData_2_2.accRadial(:, 2));
grid on; grid minor;
xlabel ("Flight Time (Normalized by Apogee)");
ylabel ("Acceleration (g's)");
title ("Radial Acceleration Until Apogee");
legend(["Flight 1", "Flight 2"], 'Location','best');
xlim([0,1])
PSPStyler(f, p, 'Light');

radialAccel_1_power = abs(ifft(ravenData_1_1.accRadial(1:6441, 2))) .^2;
dt = ravenData_1_1.accRadial(35, 1) - ravenData_1_1.accRadial(34, 1);
nu = length(ravenData_1_1.accRadial(1:6441, 2));
freq_1 = fftfreq(nu, dt);
radialAccel_2_power = abs(ifft(ravenData_2_2.accRadial(1:2358, 2))) .^2;
dt = ravenData_2_2.accRadial(35, 1) - ravenData_2_2.accRadial(34, 1);
nu = length(ravenData_2_2.accRadial(1:2358, 2));
freq_2 = fftfreq(nu, dt);

f = figure();
p = plot(freq_1, radialAccel_1_power, 'o');
hold on;
grid on;
plot(freq_2, radialAccel_2_power, 'o');
xlabel ("Frequency (Hertz)");
ylabel ("Power");
title ("Radial Acceleration Dominant Frequencies");
legend(["Flight 1", "Flight 2"]);
xlim([-30, 30]);
PSPStyler(f, p, 'Light');

f = figure();
p = plot(ravenData_1_1.accAxial(:, 1), ravenData_1_1.accAxial(:, 2));
hold on; 
grid on; grid minor;
plot(ravenData_2_2.accAxial(:, 1), ravenData_2_2.accAxial(:, 2));
xlabel ("Time (seconds)");
ylabel ("Deviation from Nominal Acceleration (g's)");
title ("Nominal Flight Comparison: Acceleration");


%% Subfunctions
function c = ConversionDefinitions()

    %----------------------------------------------------------------
    % LENGTH
    %----------------------------------------------------------------

    c.FT2M  = 0.3048;           % meters per foot
    c.M2FT  = 1 / c.FT2M;       % feet per meter

    c.FT2IN = 12.0;             % inches per foot
    c.IN2FT = 1 / c.FT2IN;      % feet per inches

    c.M2IN  = c.M2FT * c.FT2IN; % inches per meter
    c.IN2M  = 1 / c.M2IN;       % meters per inch

    c.MI2FT = 5280;             % feet per mile
    c.FT2MI = 1 / c.MI2FT;      % miles per foot

    c.M2KM  = 1 / 1000;         % kilometers per meter
    c.KM2M  = 1 / c.M2KM;       % meters per kilometer

    c.M2MI  = c.M2FT * c.FT2MI; % miles per meter
    c.MI2M  = 1 / c.M2MI;       % meters per mile

    c.FT2KM = c.FT2M * c.M2KM;  % feet per kilometer
    c.KM2FT = 1 / c.FT2KM;      % kilometers per foot


    %----------------------------------------------------------------
    % ANGLE
    %----------------------------------------------------------------

    c.RAD2DEG = 180 / pi;       % degrees per radian
    c.DEG2RAD = pi / 180;       % radians per degree


    %----------------------------------------------------------------
    % MASS
    %----------------------------------------------------------------

    c.LB2KG = 0.45359237;       % kilograms per pound
    c.KG2LBM = 1 / c.LB2KG;     % pounds per kilogram
    
    
    %----------------------------------------------------------------
    % FORCE
    %----------------------------------------------------------------

    c.LBF2N = 4.4482216152605;  % Newtons per pound-force
    c.N2LBF = 1 / c.LBF2N;      % pound-force per Newton
    
    
    %----------------------------------------------------------------
    % TIME
    %----------------------------------------------------------------

    c.HR2SEC = 3600;            % seconds per hour
    c.SEC2HR = 1 / c.HR2SEC;    % hours per second
    
    %----------------------------------------------------------------
    % PRESSURE
    %----------------------------------------------------------------

    c.PA2PSI = c.N2LBF / (c.M2IN^2);   % PSI per Pascal
    c.PSI2PA = 1 / c.PA2PSI;           % Pascal per PSI
    
    %----------------------------------------------------------------
    % TEMPERATURE
    %----------------------------------------------------------------
    
    c.K2R = 9/5;                % Degrees Rankine per Kelvin
    c.R2K = 1 / c.K2R;          % Kelving per Degree Rankine
end

% Function that alters an inputted plot into official PSP colors. Call this
% function after creation of the entire plot, grid, legend, and any
% additional curves.
%
% Inputs:   fig - figure [object], obtained by calling fig = figure(...);
%           plotIn - plot [object], obtained by calling plotIn = plot(...);
% Optional inputs:
%           colorMode - choose between "Light" and "Dark" modes when
%               graphing. Defaults to "Dark". [string]

function PSPStyler(fig, plotIn, colorMode)
    %% Constants Declaration
    lineWidth = 2;
    gold = '#DAAA00'; % curve 1
    dust = '#EBD99F'; % curve 2
    aged = '#8E6F3E'; % curve 3
    darkColor = '#252526';
    steel = '#555960'; % grid
    lightColor = '#F3F0E9'; % text
    railwayGray= '#9D9795';

    %% Figure Settings
    if nargin < 3
        colorMode = "Dark";
    end
    
    figAxes = gca;
    %dimmensions = numel(axis) / 2;
    

    if strcmpi(colorMode, "light")
        figColor = lightColor;
        figAxesTitleColor = darkColor;
        figAxesColor = lightColor;
        figAxesXColor = darkColor;
        figAxesYColor = darkColor;
        figAxesZColor = darkColor;
        legendColor = darkColor;
        legendTextColor = lightColor;
        lineColors = {gold, railwayGray, darkColor};
    else
        figColor = darkColor;
        figAxesTitleColor = lightColor;
        figAxesColor = darkColor;
        figAxesXColor = lightColor;
        figAxesYColor = lightColor;
        figAxesZColor = lightColor;
        legendColor = lightColor;
        legendTextColor = darkColor;
        lineColors = {gold, dust, aged};
    end

    %% Figure Modification
    fig.Color = figColor;
    figAxes.Title.Color = figAxesTitleColor;
    figAxes.Color = figAxesColor;
    figAxes.XColor = figAxesXColor;
    figAxes.YColor = figAxesYColor;
    figAxes.ZColor = figAxesZColor;
    colororder(fig,lineColors);
    figAxes.GridColor = steel;
    figAxes.GridAlpha = 0.9;

    if ~isempty(figAxes.Legend)
        figAxes.Legend.Color = legendColor;
        figAxes.Legend.TextColor = legendTextColor;
    end
%     plotIn.set('LineWidth',lineWidth)
    set(fig, 'InvertHardCopy', 'off');
end

% function to compute frequencies
function freq = fftfreq(nu, dt)
    if mod(nu, 2) == 0
        k_vals = [(0:nu/2-1), (-nu/2:-1)];
    else
        k_vals = [(0:(nu-1)/2), (-(nu-1)/2:-1)];
    end
    freq = k_vals./(dt.*nu);
end