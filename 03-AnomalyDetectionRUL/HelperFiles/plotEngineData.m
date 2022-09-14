function plotEngineData(trainData,varNames,nsample)
% Plot nsample number of engines for each of the
% varNames variables

% Copyright 2022 The MathWorks, Inc.

    engIdx = randi(length(trainData),nsample,1); % choose nsample engines at random
    figure
    for ii = 1:length(varNames) % separate subplot for each variable
        subplot(length(varNames),1,ii)
        hold on
        for jj = 1:length(engIdx)
            if istable(trainData{engIdx(jj)})
                x = trainData{engIdx(jj)}.Time;
                y = trainData{engIdx(jj)}.(varNames{ii});
            else % used when plotting fused sensors
                y = trainData{engIdx(jj)};
                x = 1:length(y);
            end
            plot(x,y)
        end
        hold off
        ylabel(varNames{ii})
        if ii == length(varNames)
            xlabel('Time')
        end
    end
end