function [is_valid_bbox] = nms_human(bboxes, confidences, img_size)


%Truncate bounding boxes to image dimensions

bboxes(:,3)=bboxes(:,3)+bboxes(:,1);
bboxes(:,4)=bboxes(:,4)+bboxes(:,2);
num_detections = size(confidences,1);

%higher confidence detections get priority.
[confidences, ind] = sort(confidences, 'descend');
bboxes = bboxes(ind,:);

% indicator for whether each bbox will be accepted or suppressed
is_valid_bbox = logical(zeros(1,num_detections)); 

for i = 1:num_detections
    cur_bb = bboxes(i,:);
    cur_bb_is_valid = true;
    
    for j = find(is_valid_bbox)
        %compute overlap with each previously confirmed bbox.
        
        prev_bb=bboxes(j,:);
        bi=[max(cur_bb(1),prev_bb(1)) ; ... 
            max(cur_bb(2),prev_bb(2)) ; ...
            min(cur_bb(3),prev_bb(3)) ; ...
            min(cur_bb(4),prev_bb(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0                
            % compute overlap as area of intersection / area of union
            ua=(cur_bb(3)-cur_bb(1)+1)*(cur_bb(4)-cur_bb(2)+1)+...
               (prev_bb(3)-prev_bb(1)+1)*(prev_bb(4)-prev_bb(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov > 0.2 %If the less confident detection overlaps too much with the previous detection
                cur_bb_is_valid = false;
            end
            
            %special case-- the center coordinate of the current bbox is
            %inside the previous bbox.
            center_coord = [(cur_bb(1) + cur_bb(3))/2, (cur_bb(2) + cur_bb(4))/2];
            if( center_coord(1) > prev_bb(1) && center_coord(1) < prev_bb(3) && ...
                center_coord(2) > prev_bb(2) && center_coord(2) < prev_bb(4))
               
                cur_bb_is_valid = false;
            end
            
        end
    end
    
    is_valid_bbox(i) = cur_bb_is_valid;

end

%This statement returns the logical array 'is_valid_bbox' back to the order
%of the input bboxes and confidences
reverse_map(ind) = 1:num_detections;
is_valid_bbox = is_valid_bbox(reverse_map);


fprintf(' non-max suppression: %d detections to %d final bounding boxes\n', num_detections, sum(is_valid_bbox));


