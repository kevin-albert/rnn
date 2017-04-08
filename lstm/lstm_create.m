function [ W, b ] = lstm_create( num_input, num_cells )
    %
    % Weights look like:
    % +----------+
    % | Wax  Wah |
    % | Wix  Wih |
    % | Wfx  Wfh |
    % | Wox  Woh |
    % +----------+
    %
    W = randn(4*num_cells, num_input + num_cells) / sqrt(num_input + num_cells);
    
    %
    % Biases look like:
    % [ ba  bi  bf  bo ]
    % bf starts out at 5
    %
    b = randn(4*num_cells,1);
    b(2*num_cells:3*num_cells) = 5;

end

