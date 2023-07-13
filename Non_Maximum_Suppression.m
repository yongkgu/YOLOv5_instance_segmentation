function pred = Non_Maximum_Suppression(pred_in,conf_thres,iou_thres)

% Non_Maximum_Suppression 
% Select strongest multiclass bounding boxes from overlapping clusters
%
% Inputs:
% pred_in    - Inference output dlarray of pretrained YOLO v5s
%              instance segmentation dlnetwork (25200 x 117)
% conf_thres - Confidence threshold (0~1) Default: 0.25
% iou_thres  - IOU threshold (0~1) Default: 0.45
%
% Outputs:
% pred       - List of detections, on (N,38) tensor per image 
%              [xyxy, conf, cls, mask] (N x 38)

% IVL

% setup (If you need it)
% ------------------------------------------------------------------------------
% maximum number of boxes into x = x[x[:, 4].argsort(descending=True)[:max_nms]]
% max_nms = 30000; 
% seconds to quit after
% time_limit = 0.5 + 0.05 * bs;  
% require redundant detections
% redundant = true;  
%-------------------------------------------------------------------------------

% candidates logical
x = pred_in(pred_in(:,5) > conf_thres,:); 

% Compute conf
x(:,6:end) = x(:,6:end).*x(:,5); % conf = obj_conf * cls_conf

% zero columns if no masks
mask = x(:, 86:end); 

% best class only
[conf, class] = max(x(:, 6:85),[],2);

% Detections matrix nx6 (xyxy, conf, class, mask)
x = [x(:,1:4) conf class mask];

% Filter by class
% if classes is not None:
%     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

% sort by confidence and remove excess boxes
[~,idxs] = sort(x(:, 5),'descend'); 
x = x(idxs,:);

% GPU to CPU
if(canUseGPU())
    x = gather(x); 
end

% Split Data (Bounding boxes, scores, class)
allBBoxes = x(:, 1:4); 
allScores = x(:, 5);  
allLabels = x(:, 6); 
[bboxes,scores,~,index] = selectStrongestBboxMulticlass(allBBoxes,allScores,allLabels,...
    'RatioType','Min','OverlapThreshold',iou_thres); % NMS

% OUTPUT (xyxy, scores, class, mask)
pred = [bboxes scores x(index, 6) mask(index,:)]; 

end