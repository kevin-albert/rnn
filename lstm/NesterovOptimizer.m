classdef NesterovOptimizer < handle
    properties
        v
        rate
        momentum
    end
    
    methods
        function self = NesterovOptimizer(rate, momentum, w)
            self.v = cell(w);
            self.rate = rate;
            self.momentum = momentum;
            for i = 1:length(self.v)
                self.v{i} = self.v{i} * 0;
            end
        end
        
        
        function w = optimize(self, w, g)
            
            for i = 1:length(w)
                % nesterov momentum
                self.v{i} = self.momentum^2 * self.v{i} - (1+self.momentum) * self.rate * g{i};
                w{i} = w{i} + self.v{i};
            end
        end
    end
end

