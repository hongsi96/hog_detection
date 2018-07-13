function train_human(f)
    f.numBins_hog=9;
    f.numBins_lbp=59;
    f.numHorizCells=9;
    f.numVertCells=18;
    f.cellSize=8;
    f.winSize=[f.numVertCells*f.cellSize, f.numHorizCells*f.cellSize];
    f.threshold=0.2;
    
    posFiles=getImagesInDir('../../datasets/INRIAPerson/train_64x128_H96/pos/',true);
    negFiles=getImagesInDir('../../datasets/INRIAPerson/Train/neg/',true);


    if f.cascade==false
        label_train = [ones(length(posFiles), 1); zeros(2*length(negFiles), 1)];
        fileList = [posFiles, negFiles,negFiles];
        if f.feature=='HOG'    
            X_train=zeros(length(fileList),4896);
            
        else    
            X_train=zeros(length(fileList),f.numBins_lbp*162);
        end
        fprintf('training...\n');
        fprintf('Computing features for %d training data: ',length(fileList));

        for i = 1 : length(fileList)
            
            imFile=char(fileList(i));
            
            img = imread(imFile);
            if i>length(posFiles)+length(negFiles)
                [img]=rightcrop(img);
            else
                [img]=centercrop(img);
            end
            print(i)
            img=rgb2gray(img);
            
            if f.feature=='HOG'
                H_hog=extractHOGFeatures(img);
                X_train(i, :) = H_hog';
            else
                H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                X_train(i,:)=H_lbp';
            end
            
        end


        fprintf('\n');
        % Train the SVM.
        fprintf('\nTraining linear SVM classifier...\n');
        f.svm=train_svm(X_train,label_train,1.0);
        p=X_train*f.svm;
        save('model.mat','f');
        
        numRight = sum((p > 0) == label_train);

        fprintf('\nTraining accuracy: (%d / %d) %.2f%%\n', numRight, length(label_train), numRight / length(label_train) * 100.0);

    else
        label_train_3 = [ones(length(posFiles), 1); zeros(2*length(negFiles), 1)];
        label_train_2 = [ones(1000, 1); zeros(1000, 1)];
        label_train_1 = [ones(500, 1); zeros(500, 1)];
        fileList = [posFiles, negFiles, negFiles];
        if f.feature=='HOG'    
            X_train_3=zeros(length(fileList),4896);
            X_train_2=zeros(2000,4896);
            X_train_1=zeros(1000,4896);
            
        else    
            X_train_3=zeros(length(fileList),f.numBins_lbp*162);
            X_train_2=zeros(2000,f.numBins_lbp*162);
            X_train_1=zeros(1000,f.numBins_lbp*162);
        end
        fprintf('training...\n');
        fprintf('Computing descriptors for %d training windows: ',length(fileList));

        for i = 1 : length(fileList)
            
            imFile=char(fileList(i));
            
            img = imread(imFile);
            if i>length(posFiles)+length(negFiles)
                [img]=rightcrop(img);
            else
                [img]=centercrop(img);
            end

            print(i)
            img=rgb2gray(img);
            
            if i<501
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3(i, :) = H_hog';
                    X_train_2(i, :) = H_hog';
                    X_train_1(i, :) = H_hog';
                       
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3(i,:)=H_lbp';
                    X_train_2(i,:)=H_lbp';
                    X_train_1(i,:)=H_lbp';
                end


            elseif i<1001 &&i>500

                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3(i, :) = H_hog';
                    X_train_2(i, :) = H_hog';      
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3(i,:)=H_lbp';
                    X_train_2(i,:)=H_lbp';
                end

            elseif i<=length(posFiles) &&i>1000
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3(i, :) = H_hog';    
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3(i,:)=H_lbp';
                end

            elseif i>length(posFiles) &&i<=length(posFiles)+500
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3(i, :) = H_hog';
                    X_train_2(i-length(posFiles)+1000, :) = H_hog';
                    X_train_1(i-length(posFiles)+500, :) = H_hog';
                       
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3(i,:)=H_lbp';
                    X_train_2(i-length(posFiles)+1000,:)=H_lbp';
                    X_train_1(i-length(posFiles)+500,:)=H_lbp';
                end
            elseif i>length(posFiles)+500 &&i<=length(posFiles)+1000
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3(i, :) = H_hog';
                    X_train_2(i-length(posFiles)+1000, :) = H_hog';
                   
                       
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3(i,:)=H_lbp';
                    X_train_2(i-length(posFiles)+1000,:)=H_lbp';
                    
                end
            else
                if f.feature=='HOG'
                    H_hog=extractHOGFeatures(img);
                    X_train_3(i, :) = H_hog';    
                else
                    H_lbp=extractLBPFeatures(img, 'CellSize',[f.cellSize,f.cellSize]);
                    X_train_3(i,:)=H_lbp';
                end

            
            end
            
        end


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

