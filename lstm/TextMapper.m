classdef TextMapper
    
    properties
        encode
        decode
        domain
    end
    
    methods
        function self = TextMapper(text)
            self.encode = zeros(256,1);
            self.decode = [];
            for i = 1:length(text)
                for j = 1:length(text{i})
                    c = text{i}(j);
                    if self.encode(c-0) == 0
                        self.domain = length(self.decode)+1;
                        self.decode(self.domain) = c;
                        self.encode(c-0) = self.domain;
                    end
                end
            end
        end
        
        function onehot = to_onehot(self, char)
            onehot = zeros(length(self.decode),1) - 1;
            onehot(self.encode(char-0)) = 1;
        end
        
        function char = from_onehot(self, onehot)
           [~, i] = max(onehot);
           char = native2unicode(self.decode(i));
        end
        
        function str = from_onehot_seq(self, seq)
            str = '';
            for i = 1:length(seq)
               str(i) = self.from_onehot(seq{i});
            end
        end
    end
    
end

