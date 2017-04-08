function [ seq, offset, n ] = next_sequence( text, offset, n )
%NEXT_SEQUENCE Get the next subsequence from a batch of text
%   Updates the offset
%   Returns up to n+1 characters, where each sequence overlaps the last by
%   one
    if offset >= length(text)
        offset = 1;
    end
    
    if offset + n > length(text)
        n = length(text) - offset;
    end
    
    seq = text(offset:offset+n);
    offset = offset + n;
end

