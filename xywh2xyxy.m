function y = xywh2xyxy(x)
    % Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    % x1 y1 = center x y 
    y(:, 1) = x(:, 1) - x(:, 3) / 2;  % top left x
    y(:, 2) = x(:, 2) - x(:, 4) / 2;  % top left y
    y(:, 3) = x(:, 1) + x(:, 3) / 2;  % bottom right x
    y(:, 4) = x(:, 2) + x(:, 4) / 2;  % bottom right y
  
end