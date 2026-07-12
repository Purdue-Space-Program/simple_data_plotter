%% Purdue Space Program - Liquids
%% Tango Zulu Package
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
    plotIn.set('LineWidth',lineWidth)

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
    
    set(fig, 'InvertHardCopy', 'off');
end