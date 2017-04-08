function [ data, data_length ] = read_file( filename )
    fp = fopen(filename);
    data = textscan(fp, '%s', 'Delimiter', '', 'EndOfLine', '|');
    data = data{1};
    fclose(fp);
    data_length = 0;
    for i = 1:length(data)
        data_length = data_length + length(data{i});
    end
end