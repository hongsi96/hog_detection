function [image] = centercrop(img,f)
%centercrop 128, 64

[r, c, z]=size(img);



start_r=floor((r-f.cellSize*f.numVertCells)/2);
start_c=floor((c-f.cellSize*f.numHorizCells)/2);

image=img(start_r+1:start_r+f.cellSize*f.numVertCells, start_c+1:start_c+f.cellSize*f.numHorizCells,:);

end

