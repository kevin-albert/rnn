function [ out ] = lsig( x )
     out = (1 + exp(-x)) .^ -1; 
end