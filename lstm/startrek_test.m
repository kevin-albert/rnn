load LSTM.mat W

% Load training data so we know which input / output classes to use
[ files, bytes ] = read_file('startrek_data.txt');
mapper = TextMapper(files);

% XXX save this variable...
num_cells = 200;
steps = 200;

c = '^';

S = {
    lstm_state(num_cells),...
    lstm_state(num_cells),...
    lstm_state(num_cells)
};
[ Wh1, bh1, Wh2, bh2, Wh3, bh3, Wy, by ] = W{:};
for i = 1:steps
    x = mapper.to_onehot(c);
    
    S{1} = lstm_forwardpass(Wh1, bh1, S{1}, x);
    h1 = lstm_output(S{1});

    S{2} = lstm_forwardpass(Wh2, bh2, S{2}, [x; h1]);
    h2 = lstm_output(S{2});

    S{3} = lstm_forwardpass(Wh3, bh3, S{3}, [x; h2]);
    h3 = lstm_output(S{3});

    y = tanh(W{7} * h3 + W{8});
    p = (y+1) / sum(y+1);
    c = mapper.from_dist(p);
    fprintf('%s', c);
end
