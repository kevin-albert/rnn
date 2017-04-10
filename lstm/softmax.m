function [ y ] = softmax( temp, x )
    y = exp(temp*x);
    y = y/sum(y);
end

