alpha = 0.99;           % Spectral radius of the weight matrix
density = 0.02;         % weight matrix density
size = 60;              % # of recurrent units

samples = 20;           % # of audio samples to capture
washout = 10*samples;   % Washout period
train = 50*samples;     % Training period
test = 2*samples;       % Testing period
noise = 0.00001;        % Training noise factor

[y, Fs] = audioread('01.m4a', [10001,10000+samples]);
teacher = (y(:,1)+y(:,2));
teacher = teacher / 2 / max(abs(teacher));

x = zeros(size,1);
y = 0;

Wx0 = sprand(size,size,density);
Wx0(Wx0~=0) = 2*nonzeros(Wx0)-1;
Wx = Wx0 * alpha / max(abs(eigs(Wx0)));

Wb = randn(size,1) * 0.1;



% washout 
for i = 1:washout
    x = tanh(Wx*x + Wb*y);
    y = teacher(mod(i,samples)+1);
end

% training 
M = zeros(train,size);
T = zeros(train,1);
i = 1;
for i = 1:train
    x = tanh(Wx*x + Wb*y + (2*rand(size,1)-1)*noise);
    y = teacher(mod(i,samples)+1);
    
    T(i) = y;
    M(i,:) = x;
end

% linear regression
% Wy = transpose(pseudoinverse(M) * T)
Wy = transpose(((transpose(M)*M)\transpose(M))*T);

Yexp = zeros(test,1);
Yact = zeros(test,1);
X = 1:test;

% testing
err = zeros(test,1);
for i = 1:test
    x = tanh(Wx*x + Wb*y);
    y = Wy*x;
    
    t = teacher(mod(i,samples)+1);
    err(i) = (y - t)^2;
    
    Yexp(i) = t;
    Yact(i) = y;
end


p = plot(X,Yexp,':',X,Yact,'-');
p(1).LineWidth = 2;
axis([1,test,-1,1]);
legend('Original', 'Generated');
mse = sum(err) / test;
xlabel(['MSE: ' num2str(mse, '%.5e')]);
