[ files, bytes ] = read_file('startrek_data.txt');
mapper = TextMapper(files);

num_input = mapper.domain;
num_output = mapper.domain;

seq_length = 40;
epochs = 10000;
step = 1;
% soft_temp = 0.5;

load LSTM.mat num_cells W opt last_epoch E last_file P

dW = cell(W);
for i = 1:length(dW)
    dW{i}(:) = 0;
end

% LSTM states
S = cell(3, seq_length+1);
for i = 1:seq_length+1
   S{1,i} = lstm_state(num_cells); 
   S{2,i} = lstm_state(num_cells); 
   S{3,i} = lstm_state(num_cells); 
end

Y = cell(seq_length);
for i = 1:seq_length
   Y{i} = zeros(num_output,1);
end

for epoch = last_epoch:epochs
    
    % Accumulate average error for this epoch
    err = 0;
    
    % Reset state
    S{1,1} = lstm_state(num_cells);
    S{2,1} = lstm_state(num_cells);
    S{3,1} = lstm_state(num_cells);

    for idx = last_file:length(files)
        file = files{P(idx)};
        fprintf('reading file %d\n', idx);
        
        sample_start = 0;
        sample_end = 0;
        if mod(epoch, step) == 0 && idx == 1
            sample_start = randi(max(1, length(file) - 4*seq_length));
            sample_end = sample_start + 2*seq_length;
        end

        offset = 1;
        while offset < length(file)
            [ seq, offset, n ] = next_sequence(file, offset, seq_length);
            [ Wh1, bh1, Wh2, bh2, Wh3, bh3, Wy, by ] = W{:};

            % BPTT - forward
            for i = 1:n
                x = mapper.to_onehot(seq(i));
                y_ = mapper.to_onehot(seq(i+1));

                S{1,i+1} = lstm_forwardpass(Wh1, bh1, S{1,i}, x);
                h1 = lstm_output(S{1,i+1});

                S{2,i+1} = lstm_forwardpass(Wh2, bh2, S{2,i}, [x; h1]);
                h2 = lstm_output(S{2,i+1});

                S{3,i+1} = lstm_forwardpass(Wh3, bh3, S{3,i}, [x; h2]);
                h3 = lstm_output(S{3,i+1});

%                 y = softmax(soft_temp, Wy * h3 + by);
                y = tanh(Wy * h3 + by);

                err = err + sum((y-y_).^2);
                Y{i} = y;
            end

            % BPTT - backward

            d1 = 0;
            d2 = 0;
            d3 = 0;

            for i = n:-1:1
                x = mapper.to_onehot(seq(i));
                y_ = mapper.to_onehot(seq(i+1));

                % dy
                dy = Y{i} - y_;
                dh3 = transpose(Wy) * dy;

                h3 = lstm_output(S{3,i+1});
                h2 = lstm_output(S{2,i+1});
                h1 = lstm_output(S{1,i+1});

                dW{7} = dW{7} + dy * transpose(h3);
                dW{8} = dW{8} + dy;

                [ dh2, dWh3, d3 ] = lstm_backwardpass([x; h2], S{3,i+1}, Wh3,                  dh3, d3, S{3,i});
                dW{5} = dW{5} + dWh3;
                dW{6} = dW{6} + d3(1:4*num_cells);

                [ dh1, dWh2, d2 ] = lstm_backwardpass([x; h1], S{2,i+1}, Wh2, dh2(num_input+1:end), d2, S{2,i});
                dW{3} = dW{3} + dWh2;
                dW{4} = dW{4} + d2(1:4*num_cells);

                [ ~,   dWh1, d1 ] = lstm_backwardpass(      x, S{1,i+1}, Wh1, dh1(num_input+1:end), d1, S{1,i});
                dW{1} = dW{1} + dWh1;
                dW{2} = dW{2} + d1(1:4*num_cells);
            end

            % Print out some sample text
            if offset > sample_start && offset < sample_end
               fprintf(' %5d | %s\n', epoch, mapper.from_onehot_seq(Y)); 
            end


            % Optimize
            W = opt.optimize(W, clip_gradients(dW, 5));

            % Reset
            for i = 1:length(dW)
                dW{i}(:) = 0;
            end

            % end state becomes next start state
            S{1,1} = S{1,n+1};
            S{2,1} = S{2,n+1};
            S{3,1} = S{3,n+1};
        end
        
        last_file = idx + 1;
        save LSTM.mat num_cells W opt last_epoch E last_file P
    end
    
    % Average error
    E(epoch) = err / bytes;
    last_epoch = epoch;
    
    
    plot(E(1:epoch));
    drawnow
    
    last_file = 1;
    P = randperm(1:length(files));
    
    save LSTM.mat num_cells W opt last_epoch E last_file P
end
