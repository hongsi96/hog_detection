function [image] = rightcrop(img)
%centercrop 128, 64

[r, c, z]=size(img);


start_r=floor((r-96)/2);
start_c=floor((c-48)/2);

image=img(start_r+1:start_r+96, start_c+49:start_c+96,:);
image=imresize(image,3/2);

end

