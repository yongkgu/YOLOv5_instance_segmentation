function bboxes = scale_boxes(bboxes,scaledX,scaledY)

% scale_boxes
% Rescale boxes (xyxy) for visualization in MATLAB 
%
% Inputs:
% bboxes       - Bounding boxes after NMS
% scaledX      - Hidth of Model inputsize./Hidth of original image 
% scaledY      - Width of model inputsize./Width of original image 
% 
% Outputs:
% bboxes   - Post-processed mask (H W N)               

% IVL
bboxes(:,1) = bboxes(:,1) -bboxes(:,3)/2;   % top left x
bboxes(:,2) = bboxes(:,2) -bboxes(:,4)/2;   % top left y
bboxes(:,[1,3]) = bboxes(:,[1,3])./scaledX; % x_center,width
bboxes(:,[2,4]) = bboxes(:,[2,4])./scaledY; % y_center,height

end