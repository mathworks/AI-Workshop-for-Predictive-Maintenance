%[text] # Fault Detection in 5 lines of code
%[text] <u>References</u>:
%[text] 1. Verma, Nishchal K., Rahul Kumar Sevakula, Sonal Dixit, and Al Salour. “Intelligent Condition Based Monitoring Using Acoustic Signals for Air Compressors.” IEEE Transactions on Reliability 65, no. 1 (March 2016): 291–309. [https://doi.org/10.1109/TR.2015.2459684](https://doi.org/10.1109/TR.2015.2459684). \
%%
%[text] ## Load Precomputed Features
%[text] This example shows how to classify faults in acoustic recordings of air compressors using precomputed features and Machine Learning methods. 
%[text] You may [learn more](https://www.mathworks.com/help/wavelet/ug/fault-detection-using-wavelet-scattering-and-recurrent-deep-networks.html) about how the features were computed using Wavelet Scattering techniques. 
%[text] Load the feature data and view it in the workspace. The data contains 330 different features, with a "`State`" label that represents one of 8 types of air compressor faults.
load airCompressorData.mat
%%
%[text] ## Train Machine Learning Model
%[text] There are many types of machine learning models that can be used for classification. You can read about a few of them [here](https://www.mathworks.com/help/releases/R2025b/stats/supervised-learning-machine-learning-workflow-and-algorithms.html#bswlxht).
%[text] Below, write code to train a classification model on `trainData` using a `fitc*` method (`ResponseVarName = "State"`).
%[text] `Hint: help fitctree`
%[ADD CODE HERE]
%%
%[text] ## Test Machine Learning Model
%[text] Next, use the `predict` method to test the trained Machine Learning model with the `testData` table.
%[ADD CODE HERE]
%%
%[text] ## Evaluate Machine Learning Model
%[text] Now, evaluate the accuracy of the machine learning model by running the code below:
testAccuracy = sum(testData.State == YPred)/numel(YPred)*100
testAccuracy = sum(testData.State == YPred)/numel(YPred)*100
%[text] Then create a confusion matrix chart for the classification problem to view the accuracy across each fault class.
%[text] `Hint: help confusionchart`
%[ADD CODE HERE]
%%
%[text] *Copyright 2025 The MathWorks, Inc.*

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":40}
%---
