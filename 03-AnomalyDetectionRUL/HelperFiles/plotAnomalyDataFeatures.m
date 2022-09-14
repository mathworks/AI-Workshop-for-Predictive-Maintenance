function plotAnomalyDataFeatures(dataFeatures, numFeatures, isUnit,anomalyOCSVM, anomalyIF)

% Copyright 2022 The MathWorks, Inc.

    figure;
    t1 = tiledlayout(numFeatures,1,"TileSpacing","tight");
    for ii = 1:numFeatures
        t2(ii) = tiledlayout(t1,"flow","TileSpacing","compact"); %#ok<*AGROW> 
        t2(ii).Layout.Tile = ii;
        ax1 = nexttile(t2(ii));
        plot(ax1, dataFeatures{isUnit,ii}, 'o-', ...
            MarkerEdgeColor = 'red', MarkerFaceColor = 'red', ...
            MarkerSize = 5, MarkerIndices = anomalyOCSVM);
        ax2 = nexttile(t2(ii));
        plot(ax2, dataFeatures{isUnit,ii}, 's-', ...
            MarkerEdgeColor = 'red', MarkerFaceColor = 'red', ...
            MarkerSize = 6, MarkerIndices = anomalyIF);
        ax2.YTickLabel = [];
        ylabel(t2(ii),dataFeatures.Properties.VariableNames{ii},"FontSize",8);
        if ii < numFeatures
            ax1.XTickLabel = [];
            ax2.XTickLabel = [];
        end
    end
    t2(1).Children(end).Title.String = 'One-class SVM';
    t2(1).Children(end-1).Title.String = 'Isolation Forest';
end