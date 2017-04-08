function [ g ] = clip_gradients(g, lim)
    for i = 1:length(g)
       g{i} = max(-lim, min(lim, g{i}));  
    end
end