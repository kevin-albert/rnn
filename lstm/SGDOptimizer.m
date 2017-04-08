classdef SGDOptimizer

    properties
        rate
    end
    
    methods
        function self = SGDOptimizer(rate)
            self.rate = rate;
        end
        
        function w_ = optimize(self, w, g)
           w_ = w - self.rate * g;
        end
    end
    
end


