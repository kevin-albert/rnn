%
% output:
%
% dI = [ dx  dh ]
%
% dW = |-------------|
%      | dWax   dWah |
%      | dWix   dWih |
%      | dWfx   dWfh |
%      | dWox   dWoh |
%      |-------------|
%
% d = [ da  di  df  do  dc  dh(t-1) ]
%
function [ dx, dW, d ] = lstm_backwardpass( x, state, W, dh, d_next, state_prev )
    num_cells = length(state)/6;
    
    % Get state info
    a = state(1:num_cells);
    i = state(1+num_cells:2*num_cells);
    f = state(1+2*num_cells:3*num_cells);
    o = state(1+3*num_cells:4*num_cells);
    c = state(1+4*num_cells:5*num_cells);
    cp = state_prev(1+4*num_cells:5*num_cells);
    hp = state_prev(1+5*num_cells:6*num_cells);
    
    % Compute gradients
    
    % Carry over BPTT computation from t+1
    % dc(t+1) is stored in d_next
    dc = 0;
    if length(d_next) > 1
        dc = d_next(4*num_cells+1:5*num_cells) + dh .* o .* (1-tanh(c).^2);
    end
    % g'(x) = g(x)(1-g(x))
    da = (dc .* i) .* (1 - a.^2);
    di = (dc .* a) .* i .* (1-i);
    df = (dc .* cp) .* f .* (1-f);
    do = (dh .* tanh(c)) .* o .* (1-o);
    
    
    % Now dc becomes dc(t-1)
    dc = dc .* df;
    
    % Weights, dx, etc
    dz = [da; di; df; do];
    I = [x; hp];
    dW = dz * transpose(I);

    %dI = WT * dzT
    dI = transpose(W) * dz;
    dx = dI(1:length(x));
    d = [da; di; df; do; dc; dI(1+length(x):end)];
end

