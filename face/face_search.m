function [resultRects] = face_search_cascade(fs, passorfail ,scaleRange,origImg)
	resultRects = [];
	
	%windowCountImg = 0;
	num_cascade=size(fs,1);
	%passorfail=[]%ones(size(scaleRange),count(,) , );
	if size(origImg,3)>1
		origImg = rgb2gray(origImg);
	end
	
	

    fprintf('\n');
	if fs{1}.feature=='HOG'
		for k = 1: num_cascade
			windowCountImg=0;
			pass=0;
			for l = 1 : length(scaleRange)
				% Get the next scale value.
				scale = scaleRange(l);    
        		if scale <0.4
        	    	%fprintf(' scale <0.4 break\n');
        	    	break;
				end
				if k>1 && sum(sum(passorfail))==0
					break;
				end
    
				% Scale the image.
				if (scale == 1)
					img = origImg;
				else
					img = imresize(origImg, scale);
        		end
		        
				% Convert to grayscale by averaging the three color channels.
				
				[img, xoffset, yoffset,num_height_cell,num_width_cell] = crop(fs{k}, img);
        
        		H=extractHOGFeatures(img);
				
				for i = 1:num_height_cell-15+1

            		for j = 1:num_width_cell-10+1
						if passorfail(l,j+(i-1)*(num_width_cell-9))==0
							continue;
						end
						h=[];
                		start= (j-1)*36*(num_height_cell-1)+(i-1)*36;
                		for w = 1: 10-1
                    		h_part=H(start+1+(w-1)*36*(num_height_cell-1):start+36*(15-1)+(w-1)*36*(num_height_cell-1));
                    		h=[h,h_part];
                		end
                
                		p=h*fs{k}.svm;
					
				
						if (p > fs{k}.threshold)
							pass = pass + 1;
							if (k==num_cascade)
						
								passorfail(l,j+(i-1)*(num_width_cell-7))=1;
								xstart = xoffset + ((j - 1) * fs{k}.cellSize)+1;
								ystart = yoffset + ((i - 1) * fs{k}.cellSize)+1;
	
						
								origX = round(xstart / scale);
								origY = round(ystart / scale);
								origWidth = round(fs{k}.winSize(2) / scale);
								origHeight = round(fs{k}.winSize(1) / scale);
						
							% Add the rectangle to the results.
								resultRects = [resultRects; 
										   origX, origY,origWidth,origHeight, p];
                            end
                            passorfail(l,j+(i-1)*(num_width_cell-9))=1;


						else
							passorfail(l,j+(i-1)*(num_width_cell-9))=0;
						end               

						% Increment the count of windows processed.
						windowCountImg = windowCountImg + 1;
                
                
					end
			
        		end
        
        		%fprintf('%d matches total, %d done\n', pass, windowCountImg);
        
			end


			if k==1
				passorfail=floor((passorfail+1)/2);
            end
            fprintf('%d boxes are survived\n',sum(sum(passorfail)));
		end
    
    else
		
		for k = 1: num_cascade
			windowCountImg=0;
			pass=0;
			for l = 1 : length(scaleRange)
			
				scale = scaleRange(l);    
        		if scale <0.4
        	    	%fprintf(' scale <0.4 break\n');
        	    	break;
        		end
				if k>1 && sum(sum(passorfail))==0
					break;
				end
				% Scale the image.
				if (scale == 1)
					img = origImg;
				else
					img = imresize(origImg, scale);
        		end
		        
			
				[img, xoffset, yoffset,num_height_cell,num_width_cell] = crop(fs{k}, img);
        
        		H=extractLBPFeatures(img,'CellSize',[fs{k}.cellSize,fs{k}.cellSize],'Upright',true);
			
				for i = 1:num_height_cell-15+1

            		for j = 1:num_width_cell-10+1
						if passorfail(l,j+(i-1)*(num_width_cell-9))==0
							continue;
						end
						
						h=[];
                		start = (j-1)*fs{k}.numBins_lbp*num_height_cell+(i-1)*fs{k}.numBins_lbp;
                		for w = 1: 10
                    		h_part=H(start+1+(w-1)*fs{k}.numBins_lbp*num_height_cell : start+fs{k}.numBins_lbp*15+(w-1)*fs{k}.numBins_lbp*num_height_cell);
                    		h=[h,h_part];
                		end
                
					%[l,p]=predict(hog.SVMModel, h);
                	%p=[4.*rand(1,1),4.*rand(1,1)];
                	%p=[3,3];
                		p=h*fs{k}.svm;
					
						
						
						if(p>fs{k}.threshold)
							passorfail(l,j+(i-1)*(num_width_cell-9))=1;
							pass = pass + 1;
							if (k==num_cascade)
						
								passorfail(l,j+(i-1)*(num_width_cell-9))=1;
								xstart = xoffset + ((j - 1) * fs{k}.cellSize)+1;
								ystart = yoffset + ((i - 1) * fs{k}.cellSize)+1;
	
						
								origX = round(xstart / scale);
								origY = round(ystart / scale);
								origWidth = round(fs{k}.winSize(2) / scale);
								origHeight = round(fs{k}.winSize(1) / scale);
						
							% Add the rectangle to the results.
								resultRects = [resultRects; 
										   origX, origY,origWidth,origHeight, p];	

                            end

						else			
							passorfail(l,j+(i-1)*(num_width_cell-9))=0;
						end               

					% Increment the count of windows processed.
						windowCountImg = windowCountImg + 1;
                
                
					end
			
        		end
        
        		%fprintf('%d matches total, %d done\n', pass, windowCountImg);
        
			end

			if k==1
				passorfail=floor((passorfail+1)/2);
            end
            fprintf('%d boxes are survived\n',sum(sum(passorfail)));

		end
        
    end
end