function [resultRects] = human_search(fs, passorfail ,scaleRange,origImg)
	resultRects = [];
	
	%windowCountImg = 0;
	num_cascade=size(fs,1);
	%passorfail=[]%ones(size(scaleRange),count(,) , );
	origImg = rgb2gray(origImg);
	

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
				
				[img, xoffset, yoffset,num_height_cell,num_width_cell] = crop(fs{k}, img);
        
        		H=extractHOGFeatures(img);
				
				for i = 1:num_height_cell-18+1

            		for j = 1:num_width_cell-9+1
						if passorfail(l,j+(i-1)*(num_width_cell-8))==0
							continue;
						end
						h=[];
                		start= (j-1)*36*(num_height_cell-1)+(i-1)*36;
                		for w = 1: 9-1
                    		h_part=H(start+1+(w-1)*36*(num_height_cell-1):start+36*(18-1)+(w-1)*36*(num_height_cell-1));
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
                            passorfail(l,j+(i-1)*(num_width_cell-8))=1;


						else
							passorfail(l,j+(i-1)*(num_width_cell-8))=0;
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
			
				for i = 1:num_height_cell-18+1

            		for j = 1:num_width_cell-9+1
						if passorfail(l,j+(i-1)*(num_width_cell-7))==0
							continue;
						end
						
						h=[];
                		start = (j-1)*fs{k}.numBins_lbp*num_height_cell+(i-1)*fs{k}.numBins_lbp;
                		for w = 1: 9
                    		h_part=H(start+1+(w-1)*fs{k}.numBins_lbp*num_height_cell : start+fs{k}.numBins_lbp*18+(w-1)*fs{k}.numBins_lbp*num_height_cell);
                    		h=[h,h_part];
                		end
                
                		p=h*fs{k}.svm;
					
						
						
						if(p>fs{k}.threshold)
							passorfail(l,j+(i-1)*(num_width_cell-7))=1;
							pass = pass + 1;
							if (k==num_cascade)
						
								passorfail(l,j+(i-1)*(num_width_cell-7))=1;
								xstart = xoffset + ((j - 1) * fs{k}.cellSize)+1;
								ystart = yoffset + ((i - 1) * fs{k}.cellSize)+1;
	
						
								origX = round(xstart / scale);
								origY = round(ystart / scale);
								origWidth = round(fs{k}.winSize(2) / scale);
								origHeight = round(fs{k}.winSize(1) / scale);

								resultRects = [resultRects; 
										   origX, origY,origWidth,origHeight, p];	

                            end

						else			
							passorfail(l,j+(i-1)*(num_width_cell-7))=0;
						end               

						windowCountImg = windowCountImg + 1;
                
                
					end
			
        		end
        
        		fprintf('%d matches total, %d done\n', pass, windowCountImg);
        
			end

			if k==1
				passorfail=floor((passorfail+1)/2);
            end
            fprintf('%d boxes are survived\n',sum(sum(passorfail)));

		end
        
    end
end