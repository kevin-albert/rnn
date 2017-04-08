function [ h ] = lstm_output( state )
    num_cells = length(state)/6;
    h = state(1+5*num_cells:6*num_cells);
end

