function test_human(f)
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
       f.threshold=-1;
       fs={f};
    end
    
    passorfail=-ones(10,(floor(2000/8)-17)*(floor(2000/8)-8));
    % Read in the image to be searched.
    testfiles=getImagesInDir('../../datasets/INRIAPerson/Test/pos/',true);
    
    
    fprintf('computing test for %d testing images: ', length(testfiles));
    %fprintf('Feature is : ');fprintf(f.mode);fprintf('\n');
    tic();
    rect_total=[];
    im_path=[];
    scaleRange=[1];
    for i =1:10
        scaleRange=[scaleRange,scaleRange(i)/1.2];
    end
    
    for i =1 : length(testfiles)
        
       imgFile=char(testfiles(i));
       C=strsplit(imgFile,'/');
       fprintf('\niteration:');
       print(i);
       
       img=imread(imgFile);
       
       img_id=C(7);
       
       [resultRects]=human_search(fs,passorfail,scaleRange,img);
       
       
       if size(resultRects,1)>0
          [valid]=nms_human(resultRects(:,1:4),resultRects(:,5),size(img));
          valid_index=find(valid);
      
          resultRects=resultRects(valid_index,:);
       
          num=size(resultRects,1);
          rect_total=[rect_total;resultRects];
          p=cell(num,1);
          p(:)=img_id;
          im_path=[im_path;p];
       else
           fprintf('\n no detection \n');
       end
    end
    
    elapsed = toc();
    fprintf('Image search took %.2f seconds\n', elapsed);
    

    path='./human/human_gt';
    [gt_ids, gt_boxes, gt_is,tp,fp,duplicated,cur_gt_ids]=evaluate_detections(rect_total(:,1:4),rect_total(:,5),im_path,path,'false');

end

