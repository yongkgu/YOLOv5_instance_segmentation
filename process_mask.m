function finalMasks = process_mask(proto, masks_in, box, ih,iw,ground_truth)

% process_mask 
% Mask upsampling after mask crop
%
% Inputs:
% proto        - Inference output dlarray of pretrained YOLO v5s
%                instance segmentation dlnetwork (160 x 160 x 32 x 1)
% masks_in     - (N x 32) N is number of masks after NMS
% box          - Bounding boxes after NMS
% ih           - Height of original image 
% iw           - Width of original image 
% ground_truth - Ground truth of mask score
% 
% Outputs:
% finalMasks   - Post-processed mask (H W N)               

% IVL

% Proto size (HWC)
[mh, mw, c] = size(proto,[1,2,3]); 

% Remove and Permute dlarray data format of proto 
proto = permute(stripdims(proto),[3,1,2,4]);

% 1. Reshape proto (32 x 160 x 160 x 1) into (32 * 25600)
% 2. Matrix Multiplication masks_in * proto ([N x 32] * [32 x 25600] )
% 3. Reshape masks after sigmoid (N x 160 x 160)
masks = reshape(sigmoid(masks_in * reshape(proto,c,[])),[],mh,mw);
[n, h, w] = size(masks);


% Downsampling bounding boxes
box(:, 1) = box(:, 1).* (mw ./ 640); % iw
box(:, 3) = box(:, 3).* (mw ./ 640); % iw
box(:, 4) = box(:, 4).* (mh ./ 640); % ih
box(:, 2) = box(:, 2).* (mh ./ 640); % ih

% x1 y1 x2 y2 shape: (n,1,1)
x1= zeros(n,1,2);
y1= zeros(n,1,2);
x2= zeros(n,1,2);
y2= zeros(n,1,2);
 
x1(:,1,1)= box(:, 1);
y1(:,1,1)= box(:, 2);
x2(:,1,1)= box(:, 3);
y2(:,1,1)= box(:, 4);

% cols shape(1,1,w)
c = zeros(1,h,2);
c(1,:,1) = linspace(1,h,h);
% rows shape:(1,h,1)
r = zeros(1,1,w);
r(1,1,:) = linspace(1,w,w);

% "Crop" predicted masks by zeroing out everything not in the predicted bbox.
masks_crop =  masks .* ((r(1,1,:) >= x1(:,1,1)) .* (r(1,1,:) < x2(:,1,1)) .* ...
    (c(1,:,1) >= y1(:,1,1)) .* (c(1,:,1) < y2(:,1,1)));

% mask permute (N x 160 x 160) into (160 x 160 x N)
masks_crop = gather(extractdata(permute(masks_crop,[2, 3, 1])));

finalMasks = false([ih iw n]);
% Mask upsample 
for i=1:n

    finalMasks(:,:,i) = imresize(masks_crop(:,:,i),[ih,iw]...
        ,'bilinear') > ground_truth;
end
end