function dashboard(mdl,trainData,testData,engineNum)
% Inspect a residual similarity model's performance on new test data by
% animating it over time in a series of subplots.

% Copyright 2022 The MathWorks, Inc.

    % create figure and subplots
    sz = get(0,'ScreenSize')/2;
    f=figure('Position',[1,1,sz(3),sz(4)],'Color',[0.22,0.26,0.30],'Units','normalized');
    tl = tiledlayout(f,2,5,'TileSpacing','compact');
    ax1 = nexttile(tl,[1,1]);
    ax2 = nexttile(tl,[1,4]);
    ax3 = nexttile(tl,[1,1]);
    ax4 = nexttile(tl,[1,4]);

    shadowedCurveColor = [0.5,0.5,0.5];
    highlightedCurveColor = [1,1,1];
    testCurveColor = [0.07,0.62,1.00];
    rulColor = [0.8500,0.3250,0.0980];
    backgroundAxisColor = [0.22,0.26,0.30];

    % preallocate variables
    ttf = -length(testData)+1:0;
    rulAnswer = length(testData)-1:-1:0;
    estRUL = nan(length(testData),1);
    ciRUL = nan(length(testData),2);
    pdfRUL = cell(length(testData),1);
    histRUL = cell(length(testData),1);
    [estRUL(1), ciRUL(1,:), pdfRUL{1}, histRUL{1}] = predictRUL(mdl, testData(1));

    % set up fist label
    axis(ax1,'off')
    label1 = text(ax1,'Units','normalized','Position',[0.9,0.5,0],...
        'FontSize',50,'FontWeight','bold','Margin',1,...
        'Color',testCurveColor,'FontUnits','normalized',...
        'HorizontalAlignment','right');

    % set up first plot
    set(f,'CurrentAxes',ax2)
    ax2.Color = backgroundAxisColor;
    ax2.XColor = highlightedCurveColor;
    ax2.YColor = highlightedCurveColor;

    hold(ax2,'on')
    cellNum = num2cell((1:length(trainData))');
    ptdata = cell2mat(cellfun(@(x,y) [[(1:length(x))';NaN],[x;NaN],ones(length(x)+1,1)*y],trainData,cellNum,'UniformOutput',false));

    plot(ax2,ptdata(:,1),ptdata(:,2),'Color',shadowedCurveColor,'LineWidth',0.2);
    idxKNN = knnsearch(mdl,testData(1));
    idxpknn = ismember(ptdata(:,3),idxKNN);
    pknn = plot(ax2,ptdata(idxpknn,1),ptdata(idxpknn,2),'Color',highlightedCurveColor,'LineWidth',1);
    pmdata = cellfun(@(x) [length(x),x(end)],trainData(idxKNN),'UniformOutput',false);
    pmdata = cell2mat(pmdata);
    pm = plot(ax2,pmdata(:,1),pmdata(:,2),'Color',highlightedCurveColor,'Marker','*',"LineStyle","none");
    ptest = plot(ax2,testData(1),'Color',testCurveColor,'LineWidth',2);
    hold(ax2,'off')
    xlabel(ax2,'Engine Runtime (cycles)','Color',highlightedCurveColor,'Units','normalized','FontUnits','normalized')
    ylabel(ax2,'Engine Health Indicator','Color',testCurveColor,'Units','normalized','FontUnits','normalized')
    title(ax2,['K=',num2str(mdl.NumNearestNeighbors),' Nearest Training Engine Trajetories'],'Color',highlightedCurveColor,'Units','normalized','FontUnits','normalized')
    legend(ax2,'Training Engines','K Nearest Engines','Engine Failures',['Test Engine No.', num2str(engineNum)],'AutoUpdate','off',...
        'Color',backgroundAxisColor,'TextColor',highlightedCurveColor,'EdgeColor',highlightedCurveColor,'Units','normalized',...
        'Location','best');

    % set up second label (RUL)
    axis(ax3,'off')
    label2 = text(ax3,'Units','normalized','Position',[0.9,0.5,0],...
        'FontSize',50,'FontWeight','bold','FontUnits','normalized','Margin',1,...
        'Color',rulColor,'HorizontalAlignment','right');

    % set up second plot (cycles)
    ax4.Color = backgroundAxisColor;
    ax4.XColor = highlightedCurveColor;
    ax4.YColor = highlightedCurveColor;

    hold(ax4,'on')
    rulA = plot(ax4,ttf(1),rulAnswer(1),'g','LineWidth',1); % answer
    rulE = plot(ax4,ttf(1),estRUL(1),'Color',rulColor,'LineWidth',1); % estimate
    ciL = plot(ax4,ttf(1),ciRUL(1,1),'LineStyle','--','Color',highlightedCurveColor,'LineWidth',1); % lower CI bound
    ciU = plot(ax4,ttf(1),ciRUL(1,2),'LineStyle','--','Color',highlightedCurveColor,'LineWidth',1); % upper CI bound
    ciPatch = patch(ax4,[ttf(1),ttf(1)], [ciRUL(1,1),ciRUL(1,2)], highlightedCurveColor, ...
        'FaceAlpha', 0.2, 'EdgeColor', 'none'); % CI patch
    hold(ax4,'off')
    title(ax4,"Test Engine "+engineNum,'Color',highlightedCurveColor,'Units','normalized','FontUnits','normalized')
    xlabel(ax4,"Cycles Prior to Failure",'Color',highlightedCurveColor,'Units','normalized','FontUnits','normalized')
    ylabel(ax4,"Remaining Useful Life Prediction (Cycles)",'Color',rulColor,'Units','normalized','FontUnits','normalized')
    legend(ax4,[rulA,rulE,ciPatch],["Test Answer";"Model Prediction";"90% CI"],...
        'TextColor',highlightedCurveColor,'EdgeColor',highlightedCurveColor,'Units','normalized');

    % main loop
    for ii = 2:length(testData)
        % make prediction
        [estRUL(ii), ciRUL(ii,:), pdfRUL{ii}, histRUL{ii}] = predictRUL(mdl, testData(1:ii));

        % update first subplot
        idxKNN = knnsearch(mdl,testData(1:ii));
        idxpknn = ismember(ptdata(:,3),idxKNN);
        set(pknn,'XData',ptdata(idxpknn,1),'YData',ptdata(idxpknn,2))
        pmdata = cellfun(@(x) [length(x),x(end)],trainData(idxKNN),'UniformOutput',false);
        pmdata = cell2mat(pmdata);
        set(pm,'XData',pmdata(:,1),'YData',pmdata(:,2))
        set(ptest,'YData',testData(1:ii))

        % update second subplot
        set(rulA,'XData',ttf(1:ii),'YData',rulAnswer(1:ii))
        set(rulE,'XData',ttf(1:ii),'YData',estRUL(1:ii))
        set(ciL,'XData',ttf(1:ii),'YData',ciRUL(1:ii,1))
        set(ciU,'XData',ttf(1:ii),'YData',ciRUL(1:ii,2))
        set(ciPatch,'XData',[ttf(1:ii),ttf(ii:-1:1)],'YData',[ciRUL(1:ii,1)',ciRUL(ii:-1:1,2)'])
        xlim(ax4,[ttf(1),ttf(ii)])

        % update first label
        label1.String = num2str(testData(ii),'%.2f');

        % update second label
        label2.String = num2str(round(estRUL(ii),0),'%3.0f');

        % force graphics update
        drawnow
    end
end