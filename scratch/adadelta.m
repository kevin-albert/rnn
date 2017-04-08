% hyperparams
decay = 0.9;
epsilon = 0.0001;

% parameter 
x = 0;

% gradient
g = 0;

% expected value of squared gradients
Eg = 0;

% expecteed value of squared updates
Ed = 0;


% some computation + gradient calc
x = 0.5; g = -0.2;

% accumulate expected squared gradient
Eg = decay * Eg + (1-decay) * g^2;

% calculate update value
dx = -sqrt(Ex + epsilon) / sqrt(Eg + epsilon) * g;

% accumulate expected squared update
Ex = decay * Ex + (1-decay) * dx^2;

% apply update
g = x + dx; 

