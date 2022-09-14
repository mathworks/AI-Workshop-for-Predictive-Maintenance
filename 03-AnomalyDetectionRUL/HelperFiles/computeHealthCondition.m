function data = computeHealthCondition(data)
% Compute the health indicator value for each ensemble member.
% Health indicator for each member is assumed to start from 1 
% (healthy), and decreases to 0 when it fails.

% Copyright 2022 The MathWorks, Inc.

    rul = max(data.Time) - data.Time;
    data.healthCondition = rul / max(rul);
end