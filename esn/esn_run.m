function [ data, mse ] = esn_run( samples, nwsize, train, test, alpha, density, noise, reset )
    
    washout = samples;
    train = train * samples;
    test = test * samples;

    [y, ~] = audioread('01.m4a', [10001,10000+samples]);
    
    teacher = (y(:,1)+y(:,2));
    teacher = teacher / 2 / max(abs(teacher));

    x = zeros(nwsize,1);
    y = 0;

    Wx = sprand(nwsize,nwsize,density);
    Wx(Wx~=0) = 2*nonzeros(Wx)-1;
    Wx = Wx * alpha / max(abs(eigs(Wx)));

    Wb = randn(nwsize,1) * 0.1;



    % washout 
    for i = 1:washout
        x = tanh(Wx*x + Wb*y);
        y = teacher(mod(i,samples)+1);
    end
    
    x0 = x;

    % training 
    M = zeros(train,nwsize);
    T = zeros(train,1);
    for i = 1:train
        x = tanh(Wx*x + Wb*y + (2*rand(nwsize,1)-1)*noise);
        y = teacher(mod(i,samples)+1);

        T(i) = y;
        M(i,:) = x;
    end

    % linear regression
    % Wy = transpose(pseudoinverse(M) * T)
    Wy = transpose(((transpose(M)*M)\transpose(M))*T);

    data = zeros(test,3);
    
    % testing
    if reset == 1 
        x = x0;
    end
    
    err = zeros(test,1);
    for i = 1:test
        x = tanh(Wx*x + Wb*y);
        y = Wy*x;

        t = teacher(mod(i,samples)+1);
        err(i,1) = (y - t)^2;

        data(i,:) = [i,y,t];
    end
    
    mse = sum(err,1)/test;
end

