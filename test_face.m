function test_face(f)
    addpath('./human/');
    addpath('./utils/');
    if f.cascade==true
        f1=f;
        f1.svm=f.svm_1;
        f1.threshold=-0.1;
        f2=f;
        f2.svm=f.svm_2;
        f2.threshold=0;
        f3=f;
        f3.svm=f.svm_3;
        f3.threshold=0.1;
       fs={f1;f2;f3};
       
    else
       f.threshold=0.6;
       fs={f};
    end

    passorfail=-ones(10,(floor(2000/8)-14)*(floor(2000/8)-9));
    %test file
    test_file_1='../../datasets/fddb/FDDB-folds/FDDB-fold-09-ellipseList.txt';
    test_file_2='../../datasets/fddb/FDDB-folds/FDDB-fold-10-ellipseList.txt';
    test_file={test_file_1,test_file_2};  



    %raw_test_list
    raw_test_list=cell(0,1);
    for i = 1:size(test_file,2)
        fid=fopen(test_file{1,i});
        line_ex=fgetl(fid);
        %lines=cell(0,1);
        while ischar(line_ex)
            raw_test_list{end+1,1}=line_ex;
            line_ex=fgetl(fid);
        end
    end

    %image_list
    image_list=zeros(size(raw_test_list,1),1);
    for i =1:size(raw_test_list,1)
        C=strsplit(raw_test_list{i,1},'/');
        if size(C,2)>1
            image_list(i,1)=1;
        end
    end
    %image_num
    img_num=size(find(image_list),1);
    image_index=find(image_list);

    fprintf('computing test for %d testing images: ', img_num);



    tic();
    rect_total=[];
    im_path=[];
    scaleRange=[1];
    for i =1:10
        scaleRange=[scaleRange,scaleRange(i)/1.2];
    end


    for i =1 : img_num%length(testfiles)
    
        img_id=raw_test_list{image_index(i),1};
       
       path=strcat('../../datasets/fddb/',img_id,'.jpg');
       img=imread(path);
       fprintf('\niteration:');print(i);
       
       
        [resultRects]=face_search(fs,passorfail,scaleRange,img);
       
       
       if size(resultRects,1)>0
          valid=nms_face(resultRects(:,1:4),resultRects(:,5),size(img));
          valid_index=find(valid);
      
          resultRects=resultRects(valid_index,:);
       
          num=size(resultRects,1);
          rect_total=[rect_total;resultRects];
          p=cell(num,1);
          p(:)={img_id};
          im_path=[im_path;p];
       else
           fprintf('\n no detection \n');
       end
    end
    
    elapsed = toc();
    fprintf('Image search took %.2f seconds\n', elapsed);
    
    %%
    % Validate the search results.
    
    % Load the annotations file.
    path='./face/face_gt';
    [gt_ids, gt_boxes, gt_is,tp,fp,duplicated,cur_gt_ids]=evaluate_detections(rect_total(:,1:4),rect_total(:,5),im_path,path,'false');









    

end

