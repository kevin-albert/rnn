data = 'Hello RNN';
mapper = TextMapper({data});

rate = 0.1;
momentum = 0.9;

num_input = mapper.domain;
num_cells = 9;
num_output = mapper.domain;

seq_length = length(data);
epochs = 50;
step = 10;

W = {};
[W{1}, W{2}] = lstm_create(num_input, num_cells);

W{3} = randn(num_output, num_cells) / sqrt(num_cells);
W{4} = randn(num_output, 1);

dW = cell(W);
for i = 1:length(dW)
   dW{i}(:) = 0; 
end


E = zeros(epochs,1);
for epoch = 1:epochs
    
    state = lstm_state(num_cells);
    opt = NesterovOptimizer(rate, momentum, W);
    
    sample_start = 0;
    sample_end = 0;
    if mod(epoch, step) == 0
        sample_start = randi(max(1,length(data) - 4*seq_length));
        sample_end = sample_start + 4*seq_length;
    end
    
    offset = 1;
    while offset < length(data)
        [ seq, offset, n ] = next_sequence(data, offset, seq_length);
        Y   = zeros(num_output, n);
        S   = zeros(length(state), n+1);
        S(:, 1) = state;
    
        % BPTT - forward
        for i = 1:n
            x = mapper.to_onehot(seq(i));
            y_ = mapper.to_onehot(seq(i+1));
            
            state = lstm_forwardpass(W{1}, W{2}, S(:,i), x);
            S(:,i+1) = state;
            h = lstm_output(state);
            
            y = tanh( W{3} * h + W{4} );
            E(epoch) = E(epoch) + sum((y-y_).^2);
            Y(:,i) = y;
        end

        % BPTT - backward
        d = 0;

        for i = n:-1:1
            x = mapper.to_onehot(seq(i));
            y_ = mapper.to_onehot(seq(i+1));

            dy = Y(:,i) - y_;
            h = lstm_output(S(:,i+1));
            dh = transpose(W{3}) * dy;
            
            [ ~, dWh_, d ] = lstm_backwardpass(x, S(:,i+1), W{1}, dh, d, S(:,i));
            dW{1} = dW{1} + dWh_;
            dW{2} = dW{2} + d(1:4*num_cells);

            dW{3} = dW{3} + dy * transpose(h);
            dW{4} = dW{4} + dy;
        end
        
        if offset > sample_start && offset < sample_end
           fprintf(' %5d | %s\n', epoch, mapper.from_onehot_seq(Y)); 
        end
        
        W = opt.optimize(W, dW);
        for i = 1:length(dW)
           dW{i}(:) = 0; 
        end
    end
    
    % Average error
    E(epoch) = E(epoch) / length(data);
end

% Testing
c = '^';
state = lstm_state(num_cells);
fprintf('================================================================================\n');
for i = 1:length(data)-1
    x = mapper.to_onehot(c);
    state = lstm_forwardpass(W{1}, W{2}, state, x);
    h = lstm_output(state);
    y = tanh( W{3} * h + W{4} );
    c = mapper.from_onehot(y);
    fprintf('%s', c);
end

fprintf('\n================================================================================\n');

plot(E);

