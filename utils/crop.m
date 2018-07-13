function [img, xoffset, yoffset,num_height_cell,num_width_cell] = crop(f, img)

% =============================
%       Crop The Image
% =============================

[imgHeight, imgWidth] = size(img);
%fprintf('\n befor cropped image size : h: %d, w: %d', imgHeight, imgWidth);
% Compute the number of cells horizontally and vertically for the image.
num_width_cell = floor(imgWidth / f.cellSize);
num_height_cell = floor(imgHeight / f.cellSize);

% Compute the new image dimensions.
newWidth = num_width_cell * f.cellSize;
newHeight = num_height_cell * f.cellSize;

%fprintf('\n new size : %d, %d', newHeight, newWidth);
% Divide the left-over pixels in half to center the crop region.
xoffset = round((imgWidth - newWidth)+1 / 2);
yoffset = round((imgHeight - newHeight)+1 / 2);

% Crop the image.
img = img(yoffset : (yoffset + newHeight - 1), xoffset : (xoffset + newWidth - 1));

%[h,w]=size(image);

%fprintf('\n croped image size : h: %d, w: %d', h,w);
end
