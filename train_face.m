function train_face(f)
    addpath('./face/');
    f.numBins_hog=9;
    f.numBins_lbp=59;
    f.numHorizCells=10;
    f.numVertCells=15;
    f.cellSize=8;
    f.winSize=[f.numVertCells*f.cellSize, f.numHorizCells*f.cellSize];
    f.threshold=0.2;

    negFiles = getImagesInDir('../../datasets/COCO/train2014/', true);

    train_file_1='../../datasets/fddb/FDDB-folds/FDDB-fold-01-ellipseList.txt';
    train_file_2='../../datasets/fddb/FDDB-folds/FDDB-fold-02-ellipseList.txt';
    train_file_3='../../datasets/fddb/FDDB-folds/FDDB-fold-03-ellipseList.txt';
    train_file_4='../../datasets/fddb/FDDB-folds/FDDB-fold-04-ellipseList.txt';
    train_file_5='../../datasets/fddb/FDDB-folds/FDDB-fold-05-ellipseList.txt';
    train_file_6='../../datasets/fddb/FDDB-folds/FDDB-fold-06-ellipseList.txt';
    train_file_7='../../datasets/fddb/FDDB-folds/FDDB-fold-07-ellipseList.txt';
    train_file_8='../../datasets/fddb/FDDB-folds/FDDB-fold-08-ellipseList.txt';
    train_file={train_file_1,train_file_2,train_file_3,train_file_4,train_file_5,train_file_6,train_file_7,train_file_8};
        %raw_train_list
    raw_train_list=cell(0,1);
    for i = 1:size(train_file,2)
        fid=fopen(train_file{1,i});
        line_ex=fgetl(fid);
        %lines=cell(0,1);
        while ischar(line_ex)
            raw_train_list{end+1,1}=line_ex;
            line_ex=fgetl(fid);
        end
    
    end

    %image_list
    image_list=zeros(size(raw_train_list,1),1);
    for i =1:size(raw_train_list,1)
        C=strsplit(raw_train_list{i,1},'/');
        if size(C,2)>1
            image_list(i,1)=1;
        end
    end



    images=raw_train_list(find(image_list)+1,1);

    %face_num
    face_num=0;
    for i = 1:size(images,1)
        face_num=face_num+str2num(images{i});
    end


    image_index=find(image_list);

    if f.cascade==false


        X_train=[];


        for i = 1 : 2000%length(negFiles)
    
            imFile=char(negFiles(i));
            img = imread(imFile);
            if size(img,1)<120 || size(img,2)<30
                continue;
            end
            [img]=crop_face(img,f);
            print(i)
            if size(img,3)>1
                img=rgb2gray(img);
            
            end
            
            if f.feature=='HOG'
                H_hog=extractHOGFeatures(img);
                X_train = [X_train;H_hog];
            else
                H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                X_train = [X_train;H_lbp];
            end
          
        end
        
        num_neg=size(X_train,1);


        for i =1:2000%size(image_index,1)

            [X_train,gt_int,im]=extract_feature(raw_train_list, image_index(i),X_train,f);
            print(i);
        end
        num_pos=size(X_train,1)-num_neg;

        label_train = [zeros(num_neg, 1); ones(num_pos, 1)];
        
        
        
        fprintf('\n');
        
        
        % Train the SVM.
        fprintf('\nTraining linear SVM classifier...\n');
        
        X_train=double(X_train);%rand([1396,4596]);
        
        f.svm=train_svm(X_train,label_train,1.0);
        p=X_train*f.svm;
        save('model.mat','f');
        
        
        numRight = sum((p > 0) == label_train);
        
        fprintf('\nTraining accuracy: (%d / %d) %.2f%%\n', numRight, length(label_train), numRight / length(label_train) * 100.0);


    else   
        X_train_3=[];
        X_train_2=[];
        X_train_1=[];
        for i = 1 : 2000%length(negFiles)
    
            imFile=char(negFiles(i));
            img = imread(imFile);
            if size(img,1)<120 || size(img,2)<30
                continue;
            end
            [img]=crop_face(img,f);
            print(i)
            if size(img,3)>1
                img=rgb2gray(img);
            
            end
            

            if i<501

                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3 = [X_train_3;H_hog];
                    X_train_2 = [X_train_2;H_hog];
                    X_train_1 = [X_train_1;H_hog];
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3 = [X_train_3;H_lbp];
                    X_train_2 = [X_train_2;H_lbp];
                    X_train_1 = [X_train_1;H_lbp];
                end
            elseif i<1001 && i>500
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3 = [X_train_3;H_hog];
                    X_train_2 = [X_train_2;H_hog];
                    
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3 = [X_train_3;H_lbp];
                    X_train_2 = [X_train_2;H_lbp];
                    
                end
            else
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3 = [X_train_3;H_hog];
                    
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3 = [X_train_3;H_lbp];
                    
                end
          
            end
        end
        num_neg_3=2000;
        num_neg_2=1000;
        num_neg_1=500;

        for i =1:2000%size(image_index,1)
            X_train=[];
            [X_train,gt_int,im]=extract_feature(raw_train_list, image_index(i),X_train,f);
            if i<501
                X_train_3=[X_train_3;X_train];
                X_train_2=[X_train_2;X_train];
                X_train_1=[X_train_1;X_train];
            elseif i<1001
                X_train_3=[X_train_3;X_train];
                X_train_2=[X_train_2;X_train];
            else
                X_train_3=[X_train_3;X_train];
            end
            print(i);
        end



        num_pos_3=size(X_train_3,1)-num_neg_3;
        num_pos_2=size(X_train_2,1)-num_neg_2;
        num_pos_1=size(X_train_1,1)-num_neg_1;
        label_train_3 = [zeros(num_neg_3, 1); ones(num_pos_3, 1)];
        label_train_2 = [zeros(num_neg_2, 1); ones(num_pos_2, 1)];
        label_train_1 = [zeros(num_neg_1, 1); ones(num_pos_1, 1)];






        X_train_3=double(X_train_3);
        X_train_2=double(X_train_2);
        X_train_1=double(X_train_1);
        fprintf('\n');
        % Train the SVM.
        fprintf('\nTraining linear SVM classifier...\n');
        f.svm_3=train_svm(X_train_3,label_train_3,1.0);
        p_3=X_train_3*f.svm_3;
    
        f.svm_2=train_svm(X_train_2,label_train_2,1.0);
        p_2=X_train_2*f.svm_2;

        f.svm_1=train_svm(X_train_1,label_train_1,1.0);
        p_1=X_train_1*f.svm_1;

        save('model.mat','f');
        
        numRight_3 = sum((p_3 > 0) == label_train_3);
        numRight_2 = sum((p_2 > 0) == label_train_2);
        numRight_1 = sum((p_1 > 0) == label_train_1);
        fprintf('\nThird SVM Training accuracy: (%d / %d) %.2f%%\n', numRight_3, length(label_train_3), numRight_3 / length(label_train_3) * 100.0);
        fprintf('\nSecond SVM Training accuracy: (%d / %d) %.2f%%\n', numRight_2, length(label_train_2), numRight_2 / length(label_train_2) * 100.0);
        fprintf('\nFirst SVM Training accuracy: (%d / %d) %.2f%%\n', numRight_1, length(label_train_1), numRight_1 / length(label_train_1) * 100.0);        

    end


end

