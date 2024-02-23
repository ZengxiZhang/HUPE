function [Grad] = getGrad(I, win)
%取y通道后，用y通道的图像在win*win的窗口下取该窗口内最大值，后减去自己的平均像素值后再除以（最大像素值-最小像素值）
r0 = 0;
r1 = 1;
Strch = @(x) ( x-min(x(:)) ).*( (r1-r0) / ( max(x(:)) - min(x(:)) ) ) + r0; % @表示命名了一个名为Strch的函数
I = im2double( I );
imYUV = rgb2ycbcr( I );
[Gmag, ~] = imgradient( imYUV(:,:,1), 'sobel' ); % y通道提取梯度

radius = win;
se = strel('square', radius);

imG = imdilate(Gmag, se);% 应是规定一个win正方窗口大小的矩形模板，对Gmag进行掩膜，每个值取邻域radius * radius里最大的值。
Grad = Strch(imfill(imG, 8, 'holes'));
