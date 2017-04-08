function [ state1 ] = lstm_forwardpass( W, b, state, x )
    num_cells = length(state)/6;
    state1 = lstm_state(num_cells);
    
    cp = state(1+4*num_cells:5*num_cells);
    h = state(1+5*num_cells:6*num_cells);
  
    % Calc all gates, input values
    %z = W * transpose([x, h]) + b;
    z = W*[x;h] + b;
    
    a = tanh(z(1:num_cells));
    i = lsig(z(1+num_cells:2*num_cells));
    f = lsig(z(1+2*num_cells:3*num_cells));
    o = lsig(z(1+3*num_cells:4*num_cells));
    
    % Calc cell state
    % note - cell state is linear (not tanh'd)
    c = a .* i + cp .* f;
    
    state1(1:num_cells) = a;                          % a
    state1(1+num_cells:2*num_cells) = i;              % i
    state1(1+2*num_cells:3*num_cells) = f;            % f
    state1(1+3*num_cells:4*num_cells) = o;            % o
    state1(1+4*num_cells:5*num_cells) = c;            % c
    state1(1+5*num_cells:6*num_cells) = tanh(c) .* o; % h
end

