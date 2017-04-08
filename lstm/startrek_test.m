load LSTM.mat W

steps = 200;

c = '^';

S = {
    lstm_state(num_cells),...
    lstm_state(num_cells),...
    lstm_state(num_cells)
};
for i = 1:steps
    x = mapper.to_onehot(c);
    
    S{1} = lstm_forwardpass(Wh1, bh1, S{1}, x);
    h1 = lstm_output(S{1});

    S{2} = lstm_forwardpass(Wh2, bh2, S{2}, [x; h1]);
    h2 = lstm_output(S{2});

    S{3} = lstm_forwardpass(Wh3, bh3, S{3}, [x; h2]);
    h3 = lstm_output(S{3});

    y = tanh(W{7} * h3 + W{8});
    c = mapper.from_onehot(y);
    fprintf('%s', c);
end
