rate = 0.0002;
momentum = 0.9;

[ files, bytes ] = read_file('startrek_data.txt');
mapper = TextMapper(files);

num_input = mapper.domain;
num_cells = 200;
num_output = mapper.domain;

% LSTM parameters
W = {};

% 3 LSTM layers
[ W{1}, W{2} ] = lstm_create(num_input, num_cells);
[ W{3}, W{4} ] = lstm_create(num_input+num_cells, num_cells);
[ W{5}, W{6} ] = lstm_create(num_input+num_cells, num_cells);

% FF output layer
W{7} = randn(num_output, num_cells) / sqrt(num_cells);
W{8} = randn(num_output, 1);

opt = NesterovOptimizer(rate, momentum, W);
last_epoch = 1;

E = zeros(10000, 1);

save LSTM.mat W opt last_epoch E