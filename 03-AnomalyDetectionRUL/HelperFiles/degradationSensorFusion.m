function dataFused = degradationSensorFusion(data, sensorToFuse, weights, beta)
% Combine measurements from different sensors according 
% to the weights, smooth the fused data and offset the data
% so that all the data start from 1

% Copyright 2022 The MathWorks, Inc.

    % Fuse the data according to weights
    dataToFuse = data{:, cellstr(sensorToFuse)};
    dataFusedRaw = dataToFuse*weights;

    % Smooth the fused data with EWMA
    smoothedFusedData = zeros(length(dataFusedRaw),1);
    smoothedFusedData(1) = dataFusedRaw(1);
    for ii = 2:length(smoothedFusedData)
        smoothedFusedData(ii) = ewma(smoothedFusedData(ii-1), dataFusedRaw(ii), beta);
    end

    % Offset the data to 1
    dataFused = smoothedFusedData + 1 - smoothedFusedData(1);

end

function smoothData = ewma(prev, obs, beta)
    smoothData = beta * prev + (1-beta)*obs;
end