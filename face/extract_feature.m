function [H,gt_int,crop_image] = extract_feature(raw_train_list, image_num, H,f)



image_name=raw_train_list{image_num,1};
path=strcat('../../datasets/fddb/',image_name,'.jpg');
num_face=str2num(raw_train_list{image_num+1,1});
img=imread(path);
%img=im2double(img)
gt_int=zeros(num_face,4);
for i =1:num_face
    gt=strsplit(raw_train_list{image_num+1+i,1},' ');
    gt_int(i,4)=2*str2num(gt{1,1});
    gt_int(i,3)=2*str2num(gt{1,2});

    if 3*gt_int(i,3)>2*gt_int(i,4)
        gt_int(i,4)=int64(round(3*gt_int(i,3)/2));
    else
        gt_int(i,3)=int64(round(2*gt_int(i,4)/3));
    end

    gt_int(i,1)=max(int64(str2num(gt{1,4})-round(gt_int(i,3)/2)),1);
    gt_int(i,2)=max(int64(str2num(gt{1,5})-round(gt_int(i,4)/2)),1);
    
    crop_image=img(gt_int(i,2)+1:min(gt_int(i,2)+int64(gt_int(i,4)),size(img,1)), gt_int(i,1)+1:min(gt_int(i,1)+int64(gt_int(i,3)), size(img,2)),:);
    
    if 2*size(crop_image,1)>3*size(crop_image,2)
        crop_image = imresize(crop_image,80/size(crop_image,2));
         
    else
        crop_image = imresize(crop_image,120/size(crop_image,1));
    end
    crop_image=crop_image(1:120,1:80,:);
    %feature
    if size(crop_image,3)>1
    crop_image=rgb2gray(crop_image);
    end
    
    if f.feature=='HOG'
        H_hog=extractHOGFeatures(crop_image);
        H=[H;H_hog];
    %H_lbp=extractLBPFeatures(crop_image, 'CellSize', [f.CellSize, f.CellSize])
    else
        H_lbp=extractLBPFeatures(crop_image, 'CellSize',[f.cellSize,f.cellSize]);
        H=[H;H_lbp];
    end
end


end
