%[text] %[text:anchor:T_1E00DC54] # Anomaly Detection and Remaining Useful Life (RUL) Estimation
%[text] This example shows how to build a Remaining Useful Life (RUL) estimation algorithm for an aircraft engine. We start by designing an unsupervised anomaly detection algorithm, which will indicate that one of our engines may be operating outside normal range. Then, to estimate the remaining useful life, we construct a health indicator by fusing the most trendable sensors, train a similarity model to estimate RUL, and validate the algorithm performance. 
%[text] <u>References</u>:
%[text] 1. *Saxena, Abhinav, Kai Goebel, Don Simon, and Neil Eklund. "Damage propagation modeling for aircraft engine run-to-failure simulation." In Prognostics and Health Management, 2008. PHM 2008. International Conference on, pp. 1-9. IEEE, 2008.*
%[text] 2. *Saxena, Abhinav, Kai Goebel. "Turbofan Engine Degradation Simulation Data Set." NASA Ames Prognostics Data Repository* [*https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6*](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)*, NASA Ames Research Center, Moffett Field, CA* \
%[text:tableOfContents]{"heading":"**Table of Contents**"}
%%
%[text] %[text:anchor:H_092A751C] ## Anomaly Detection
%[text] First, we will build a condition monitoring system that triggers when our system has departed from normal behavior. Once we know our system has left normal conditions and has begun degrading, we can then feed the sensor data to the similarity-based model to estimate remaining useful life (RUL).
%%
%[text] %[text:anchor:H_49A11405] ### 1. Access data
%[text] We have collected data from a fleet of 100 aircraft engines (units) and 14 sensor readings averaged over the course of each flight. The data for each engine is contained in a CSV file. We will access these files using a datastore, which lets us point to and work with a collection of files that may be too large to fit into local memory. In this case, because our dataset is small, we will read all of the data into memory.
fileLocation = matlab.desktop.editor.getActiveFilename;
dataFolder = fullfile(fileparts(fileLocation),"Data","*.csv");
ds = datastore(dataFolder);
EngineData = readall(ds);
%%
%[text] %[text:anchor:H_DEBE9F8F] ### 2. Anomaly Detection with One-Class SVM
%[text] A [One-Class Support Vector Machine (SVM)](https://www.mathworks.com/help/stats/fitcsvm.html#bt8v_1z-1) is an unsupervised learning approach that tries to separate data from the origin in the transformed high-dimensional predictor space. We'll train a One-Class SVM on our engine data, which will help us identify which engine cycles are different from normal operation.
%[text] Notice the variable named `anomalyFreq`. This value is an option for the [`fitcsvm`](https://www.mathworks.com/help/stats/fitcsvm.html) classifier function. This allows us to choose what percentage of the data is assumed to be anomalous; in this case, we'll assume 5%. In practice, this number may require some iteration -- an initial estimate may be based on past maintenance data or industry estimates. The anomalous observations will have a negative value of `scoreSVM`, and anomaly indicators are stored in the logical array `isanomalySVM`*.*
anomalyFreq = 0.05; 

% Extract only the sensor data
dataFeatures = EngineData(:,3:16);
[numobs, numvars] = size(dataFeatures);

% random number generator set for reproducibility of results
rng("default") 

% Train a One-Class SVM and compute the scores
mdlOCSVM = fitcsvm(dataFeatures, ones(numobs,1), Standardize=true, OutlierFraction=anomalyFreq);
[~, scoreSVM] = resubPredict(mdlOCSVM);
isanomalySVM = scoreSVM < 0;
%[text] Next, we can view a histogram of the anomaly scores from the trained algorithm. Any data scores below the threshold of zero will be classified as anomalies.
figure
histogram(scoreSVM, Normalization="probability");
xline(0,"k-","Threshold = 0")
title("Histogram of Anomaly Scores for OCSVM")
%[text] Check the fraction of detected anomalies in the data. `fitcsvm` trains the bias term of the SVM model so that the specified fraction of the training observations has negative scores. Therefore, the previous value is close to the specified fraction value.
sum(isanomalySVM)/numobs
%%
%[text] %[text:anchor:H_FD2762D6] ### 3. Anomaly Detection with Isolation Forest
%[text] There are other one-class algorithms we can use on unlabeled data. The [Isolation Forest](https://www.mathworks.com/help/stats/isolationforest.html#mw_12d3da87-2ecb-447f-8901-b3688a7619f9_sep_shared-Definition-IsolationForest) algorithm detects anomalies by isolating them from normal points using an ensemble of isolation trees.
%[text] The decision trees of an isolation forest isolates each observation in a leaf. These anomaly scores are based on the average path lengths over all isolation trees. How many decisions a sample passes through to get to its leaf is a measure of how complicated it was to isolate it from the others. Here, we will assume the same value of `anomalyFreq` as above: 5%.
[mdlIF, ~, scoreIF] = iforest(dataFeatures, ContaminationFraction=anomalyFreq);
isanomalyIF = isanomaly(mdlIF,dataFeatures);
%[text] As we did for one-class SVM, we can then plot the histogram of anomaly scores and check the fraction of anomalies detected.
figure
histogram(scoreIF,Normalization="probability");
xline(mdlIF.ScoreThreshold,"k-", join(["Threshold =" mdlIF.ScoreThreshold]))
title("Histogram of Anomaly Scores for Isolation Forest")
sum(isanomalyIF)/numobs
%%
%[text] %[text:anchor:H_E3619108] ### 4. Compare Anomaly Detection Techniques
%[text] Here, we load observations from several different engines and search for anomalies within that data.  Choose data for an engine to be processed by setting the value for `engineUnit`**.** Run this and the remaining sections for each engine and compare the figures. Which engine's outputs show anomalies? Where?
%[text] During your analysis, make sure to compare engines. Try comparing engine 17 with engine 31. Which engine do you think needs maintenance?
engineUnit = 31; %[control:slider:8d29]{"position":[14,16]}
numFeatures = 4;
isUnit = EngineData.Unit == engineUnit;
anomalyOCSVM = find(isanomalySVM(isUnit));
anomalyIF = find(isanomalyIF(isUnit));
plotAnomalyDataFeatures(dataFeatures,numFeatures,isUnit,anomalyOCSVM,anomalyIF)
%%
%[text] To have a better understanding of these anomaly detection techniques, you may reduce the data dimension by using t-SNE. The t-SNE method (t-distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique well suited for embedding high-dimensional data for visualization in a low-dimensional space. The goal of t-SNE is to model each high-dimensional point by a low-dimensional point in such a way that close points are modeled by nearby points and further away points are modeled by distant points with high probability. This is specially useful when we expect our data to include outliers or anomalies. Thus, the [`tsne`](https://www.mathworks.com/help/stats/tsne.html) function constructs a set of embedded points in a low-dimensional space whose relative similarities mimic those of the original high-dimensional points. The embedded points show the clustering in the original data. 
% Note this may take ~3min to run
T = tsne(dataFeatures{:,:}, Standardize=true);
%%
%[text] Plot the normal points and outliers in the reduced dimensional plots. Compare the results of the two methods: the isolation forest algorithm and One-class SVM model.
figure
tiledlayout(1,2)

ax1 = nexttile;
gscatter(T(:,1),T(:,2),isanomalySVM,"kr",".x",[],"off")
title("One-Class SVM")
legend("Normal","Anomalies");
xlim(ax1, [-80 80])
ax1.XTick = [-80 -50 0 50 80];

ax2 = nexttile;
gscatter(T(:,1),T(:,2),isanomalyIF,"kr",".x",[],"off")
title("Isolation Forest")
legend("Normal","Anomalies");
xlim(ax2, [-80 80])
ax2.XTick = [-80 -50 0 50 80];
%%
%[text] The anomalies identified by the two methods are located near each other in the reduced dimensional plots. Compute the fraction of outliers that the two methods have in common, then plot the results.
sum(isanomalySVM.*isanomalyIF)/numobs
figure; 
gscatter(T(:,1),T(:,2),isanomalySVM.*isanomalyIF,"kr",".x",[],"off")
legend("Normal","Anomalies");
title("Combined Results from Both Methods")
%%
%[text] We can visualize the trajectory of a single engine through the reduced dimensional space. Each iteration represents a single flight (engine cycle) of a given engine. Notice where the engines start, and where they end up. Compare the results for a few engines. 
EngineToPlot = 55; %[control:slider:3fcc]{"position":[16,18]}
isUnit = EngineData.Unit == EngineToPlot;
figure;
gscatter(T(:,1),T(:,2),isanomalySVM.*isanomalyIF,"kr",".x",[],"off")

hold on
ptEngine = plot(nan,nan,"g-");
ptEngineLast = plot(nan,nan,"go","MarkerFaceColor","g");
legend("Normal","Anomalies","Test Engine")
x = T(isUnit,1);
y = T(isUnit,2);
for k = 1:length(x)
    set(ptEngine, "XData", x(1:k), "YData", y(1:k));
    set(ptEngineLast, "XData", x(k), "YData", y(k));
    drawnow
end
hold off
%[text] We could now use the two trained algorithms (one-class SVM and isolation forest) in combination to create one anomaly detection algorithm. Once our algorithm detects that an engine has begun to deviate from normal, you may now want to ask the next question: How long can we continue to fly?
%%
%[text] %[text:anchor:H_53FC8997] ## Remaining Useful Life Estimation
%[text] Next, you will build a Remaining Useful Life (RUL) algorithm. To do this, you will select the top trendable features, construct a health indicator by sensor fusion, train an RUL model, and validate the performance.
%%
%[text] %[text:anchor:H_143D7513] ### 1. Access Data
%[text] We will use a preprocessed dataset for our RUL algorithm development. There are three main differences between this dataset and the one we used for anomaly detection: (1) We now have data from 218 engines instead of only 100, (2) We have already preprocessed and normalized the data, and (3) We have collected the data into a cell array with each engine stored in a cell. This data format is necessary to train our RUL algorithm.
%[text] We will also set aside 10% of the data for testing the algorithm performance.
load rulData.mat

% random number generator set for reproducibility of results
rng("default") 

cv = cvpartition(numel(rulData),'Holdout',0.1);
trainData = rulData(training(cv),:);
testData = rulData(test(cv),:);
%%
%[text] %[text:anchor:H_8DB7A7D1] ### 2. Select Top Trendable Sensors
%[text] All engines in this dataset begin in a healthy state and end in failure. If we wish to tell these two states apart, we need sensor measurements that show a clear change in value from start to finish. We then select the most trendable sensors using the trendability metric: [`trendability`](https://www.mathworks.com/help/releases/R2022a/predmaint/ref/trendability.html) is the measure of similarity between the trajectories of a feature measured in several run-to-failure experiments. A more trendable feature has trajectories with the same underlying shape, which will make them more reliable condition indicators for calculating remaining useful life.
%[text] We will select the top three trendable sensors to train our algorithm. This is more robust than using a single sensor, but you can explore what happens to the algorithm with a different number of sensor inputs.
numsensors = 3;  % most trendable sensors to select %[control:dropdown:0898]{"position":[14,15]}

Trend = trendability(trainData,"Time");
Trend = rows2vars(Trend); % Transpose table
Trend = renamevars(Trend,["OriginalVariableNames","Var1"],["Sensor","Trendability"]);
TopTrending = topkrows(Trend,numsensors,"Trendability")
trendedNames = trainData{1}.Properties.VariableNames(TopTrending.Sensor)
%[text] Visualize the top trendable sensor measurements from 10 engines. 
nsample = 10;
plotEngineData(trainData,TopTrending.Sensor(1:numsensors),nsample)
%%
%[text] %[text:anchor:H_6FFAF804] ### 3. Create Ideal Health Indicator
%[text] We will be using a [similarity RUL model](https://www.mathworks.com/help/predmaint/ug/similarity-based-remaining-useful-life-estimation.html), which can accept any number of time series inputs. There are a few ways to approach this. We could:
%[text] 1. Provide all three trendable sensors directly to the similarity model.
%[text] 2. Build separate similarity models based on each trendable sensor, and use whichever model produces the best results.
%[text] 3. Fuse the trendable sensors together into a one health indicator, and build a single similarity model based on that. \
%[text] Any of these approaches could work well on a given problem, but fusing sensors tends to reduce variability and provide a more predictable signal. Furthermore, if we use the raw data directly, this can result in a very large and computationally expensive model. This could be problematic if we want to run our data in real time on an edge device.
%[text] We will use method (3) in this example: fuse the sensor measurements into a single health indicator, then build our similarity model using the health indicator as the input.
%[text] To do this, we map the top three trending sensors to a common health metric: all the data is assumed to start with a healthy condition and end in failure. The health condition at the beginning is assigned a value of 1 and the health condition at failure is assigned a value of 0. The health condition is assumed to be linearly degrading from 1 to 0 over time. We then fit a linear regression model using the sensor data as inputs and the health condition as outputs. Having done this, we can use the regression weights reconstruct this health indicator from new sensor data, and pass this into our model to predict the remaining useful life of the engine as it ages.
%[text] Compute the health condition for each individual engine, then visualize the mapped health condition. The health condition of all ensemble members change from 1 to 0 with varying degrading speeds.
trainDataHealth = cellfun(@computeHealthCondition, trainData, 'UniformOutput', false);
plotEngineData(trainDataHealth,{'healthCondition'},nsample)
%%
%[text] %[text:anchor:H_43195A48] ### 4. Create Fused Health Indicator
%[text] In the prior section we defined what ideal health indicator behavior should look like. Now, we fuse the trendable sensors together to reconstruct the ideal health indicator as closely as possible using real data. We'll do this using a simple linear regression, with the ideal health indicator as the output, and the trendable sensors as the regressors (inputs).
useTrendAnalysis = true; %[control:checkbox:6737]{"position":[20,24]}

trainDataHealthUnwrap = vertcat(trainDataHealth{:});
varNames = trainDataHealthUnwrap.Properties.VariableNames(3:end);
   
X = trainDataHealthUnwrap{:, cellstr(trendedNames)};
y = trainDataHealthUnwrap.healthCondition;
regModel = fitlm(X,y);
bias = regModel.Coefficients.Estimate(1)
weights = regModel.Coefficients.Estimate(2:end)
%[text] In order to be able to successfuly train an AI model to predict the remaining useful life, we will need to combine the measurements from the different sensors according to the obtained weights to construct a single health indicator. Additionally, it is convenient to smooth the resulting fused data, as having a very noisy health indicator can lead to poor results. You can learn how the sensor fusion and the smoothing is done by inspecting the `degradationSensorFusion` function. As for the smoothing technique, a Exponentially Weighted Moving Average is used. This is a fast, low-memory cost smooting technique, where the free parameter `beta` can help you determine how important the previous value is, the trend (${\\textrm{EWMA}}\_{t-1}${"editStyle":"visual"}), versus the most recent observation ($\\left.S\_t \\right)${"editStyle":"visual"}. Lower values of `beta` will allow to adapt more quickly to changes in the system dynamics, whereas large values of `beta` will produce smoother curves and adapt slower, being the trend more important. 
%[text] The computed value ${\\textrm{EWMA}}\_t${"editStyle":"visual"} corresponds to averaging over $\\approx \\frac{1}{1-\\beta }${"editStyle":"visual"} observations.
%[text] ${\\textrm{EWMA}}\_t =\\beta \\;{\\textrm{EWMA}}\_{t-1} +\\left(1-\\beta \\right)\\;S\_t${"editStyle":"visual"}
beta = 0.8;
trainDataFused = cellfun(@(data) degradationSensorFusion(data, trendedNames, weights, beta), trainDataHealth, ...
    'UniformOutput', false);
%%
%[text] %[text:anchor:H_8BBDD202] ### 5. Visualize the Fused Health Indicator
%[text] Let's visualize the fused health indicator we constructed in the previous section
plotEngineData(trainDataFused,{'Health Indicator'},nsample)
%[text] The Health Indicator signals have been vertically shifted so they all begin at 1. We can do this because we are assuming they are all beginning at the same healthy state, and because shifting the signal vertically does not change its shape. Notice that the condition does not always reach zero. This is expected because the health indicator is based on real, noisy sensor data. We care about the shape of the condition indicator.
%[text] You might consider stretching the training data so that it ends at 0 in addition to beginning at 1, but we do not do so for two reasons:
%[text] 1. The model we use does not require this. As is explained in more detail below, the model tries to determine how much longer a new engine will run by looking for similarly shaped trajectories in the training data and measuring their length. 
%[text] 2. Stretching the training data would deform the shape of the curve differently for each engine. When we aquire data from new engines with unknown failure times, it would be impossible to know how to deform the new data to make the comparison fair.  \
%%
%[text] %[text:anchor:H_3E5EE24F] ### 6. Train a Similarity RUL Model
%[text] Now we will build a residual similarity RUL model using the training data. This works similarly to a traditional KNN machine learning model, but it compares entire timeseries to each other rather than individual points. During model training, the model simplifies the training data by replacing each engine's Health Indicator timeseries with a simple second order polynomial model fit to that data -- this requires storing only three parameters per training example rather than hundreds of raw data points. This drastically reduces the amount of data we need to store in the model, making it more portable for embedded deployment if needed.
K = 10;
mdl = residualSimilarityModel(...
    'Method', 'poly2',...
    'Distance', 'absolute',...
    'NumNearestNeighbors', K,...
    'Standardize', 1);

fit(mdl, trainDataFused);
%%
%[text] %[text:anchor:H_10165A4A] ### 7. Test Model on New Data
%[text] Estimate RUL for a test engine and compare the results with the actual life remaining. We can choose any test engine and start out by feeding only the first few datapoints into the model. This represents how the model would estimate RUL at the beginning of engine operation. We then add test data gradually to see if the RUL Estimation improves as it receives more data.
%[text] Now that we have the trained model, let's update our test data with the same preprocessing step.
testDataHealth = cellfun(@computeHealthCondition, testData, 'UniformOutput', false);
testDataFused = cellfun(@(data) degradationSensorFusion(data, trendedNames, weights, beta), testDataHealth, ...
    'UniformOutput', false);
%%
%[text] For a given engine, try adjusting `CycleNumber` to see how the estimated RUL varies as we get closer to failure. If we are only 5 cycles in to the life of the engine, how accurately do you expect the RUL prediction to be? Compare `actualRUL` to `estRUL`.
EngineNum = 8; %[control:dropdown:0cc3]{"position":[13,14]}
CycleNumber = 100; %[control:dropdown:0ee3]{"position":[15,18]}

actualRUL = height(testData{EngineNum,1})-CycleNumber
estRUL = predictRUL(mdl,testDataFused{EngineNum,1}(1:CycleNumber)) 
error = actualRUL - estRUL
%%
%[text] Comparing RUL error for all engines at the given `CycleNumber`.
CycleNumber = 100; %[control:dropdown:12c2]{"position":[15,18]}

allActualRUL = cellfun(@(data) height(data) - CycleNumber, testData);
allEstRUL = cellfun(@(data) predictRUL(mdl, data(1:CycleNumber)), testDataFused);
allError = allActualRUL - allEstRUL;
figure;
h = histogram(allError,10);
xlim([-200,200])
title("RUL error at given CycleNumber")
%[text] Identifying bad engine.
[errorBadEngine, idxBadEngine] = max(h.Data);
binEdges = h.BinEdges;
[~, binErrorBadEngine] = min(abs(binEdges - errorBadEngine));
xLoc = binEdges(1:end-1)+diff(binEdges)/2;
xLoc = xLoc(binErrorBadEngine-1);
yLoc = 1;
hold on;
text(xLoc, yLoc, num2str(idxBadEngine), "VerticalAlignment", "bottom", "HorizontalAlignment", "center", "Color", "red");
%%
%[text] %[text:anchor:H_3BFF3662] ### 8. Evaluate Results
%[text] When the model is used with a new test engine, it will find the nearest `K` ensemble members in the training data set. There are many ways to measure distance and determine who is closest. In this case the cumulative absolute distance between the test signal and the polynomial fits created during training are used. Based on that distance, `d`, the model then computes a similarity metric, $e^{-d^2 }${"editStyle":"visual"}, which only has values in the range (0,1\]. It then fits a ksdensity distribution to the similarity measures of the `K` closest training signals. It's predicted RUL is the median RUL of the distribution.
%[text] It is easiest to see how the model works by simulating it running as a new test engine generates data. We do so by running the model in a loop, where at each loop iteration we expand the length of the test data passed to the model by one data point as if that new data point was just generated. In practice, this represents collecting data after each flight.
%[text] The code in the next section creates such an animation using a dashboard-like plot. Create the animation of test engine performance. Explore the performance of different engines -- try 8 (good) and 17 (bad) as examples of very different behavior.
EngineToAnimate = 8; %[control:dropdown:0db3]{"position":[19,20]}
testDataTmp = testDataFused{EngineToAnimate};
dashboard(mdl,trainDataFused,testDataTmp,EngineToAnimate)
%[text] The animation shows two different plots as a new test engine generates data.
%[text] 1. The top plot shows all the training data trajectories in gray, the `K` training trajectories nearest to the test data in white, and the test engine trajectory in blue. When the white trajectories are broadly distributed through the training data, as is often the case when a new test engine starts, we will get a wide confidence interval. As the test engine runs and its degradation path becomes more clear, the white lines should become more concentrated and lead to a tighter confidence interval. 
%[text] 2. The bottom plot shows the correct answer, the model's RUL prediction and the 90% confidence interval plotted over time. The correct answer in green decreases linearly over time, e.g. when the test engine is 150 cycles away from failure the correct answer for its RUL is 150 cycles. This plot shows how the model's performance improves (or doesn't) as we pass larger amounts of the test engine's data to the model over time. \
%[text] This can be done for any engine, but two test engines in particular are good examples of the types of behavior the model can exhibit.
%[text] - Test engine 8 is a good example of when this type of model works well. Initially the prediction is off by about 50 cycles and the confidence interval is large at +/- 75 cycles. This should be expected since the model doesn't have much data to work with yet and it is difficult to differentiate behavior when the engine is far from failure. This can be seen in the top plot as initially the nearest training examples are widely distributed. By the time the engine is within 85 cycles of failure, our prediction is very close to the correct answer and the 90% confidence interval has tightened to +/- 20 cycles. The increase in accuracy over time is attributable to the large number of training examples whose path forms a similar shape, thus giving us many similar examples to compare against.
%[text] - Test engine 17 is an example of when this type of model will not work well. The model underpredicts RUL by a large amount and at no point does the correct answer lie within our 90% confidence interval. The reason can be seen in the top plot as the shape of its trajectory differs greatly from that of the training data. This is a good reminder that machine learning algorithms are useful for interpolating between known training examples, and not for extrapolating from known examples into unknown territory. \
%[text] %[text:anchor:H_E7F3362C] ## Summary
%[text] We have now trained two algorithms: One to detect anomalies and determine when an engine may be deviating from normal, and another to estimate the Remaining Useful Life. In practice, we could combine both of these algorithms into a deployed Predictive Maintenance system to help us identify when engine maintenance is needed. 
%%
%[text] *Copyright 2022 The MathWorks, Inc.*

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":17.3}
%---
%[control:slider:8d29]
%   data: {"defaultValue":9,"label":"engineUnit","max":100,"min":1,"run":"Section","runOn":"ValueChanging","step":1}
%---
%[control:slider:3fcc]
%   data: {"defaultValue":51,"label":"numEngine","max":100,"min":1,"run":"Section","runOn":"ValueChanged","step":1}
%---
%[control:dropdown:0898]
%   data: {"defaultValue":"3","itemLabels":["1","2","3","4","5"],"items":["1","2","3","4","5"],"label":"numsensors","run":"Section"}
%---
%[control:checkbox:6737]
%   data: {"defaultValue":false,"label":"useTrendAnalysis","run":"Section"}
%---
%[control:dropdown:0cc3]
%   data: {"defaultValue":"8","itemLabels":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"],"items":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"],"label":"Drop down","run":"Section"}
%---
%[control:dropdown:0ee3]
%   data: {"defaultValue":"100","itemLabels":["5","20","40","60","80","100","120"],"items":["5","20","40","60","80","100","120"],"label":"Drop down","run":"Section"}
%---
%[control:dropdown:12c2]
%   data: {"defaultValue":"100","itemLabels":["5","20","40","60","80","100","120"],"items":["5","20","40","60","80","100","120"],"label":"Drop down","run":"Section"}
%---
%[control:dropdown:0db3]
%   data: {"defaultValue":"8","itemLabels":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"],"items":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"],"label":"Drop down","run":"Section"}
%---
