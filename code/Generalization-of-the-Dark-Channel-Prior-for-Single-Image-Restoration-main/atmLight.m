function A = atmLight(I, DepthMap)

[height, width, ~] = size(I);
imsize = width * height;
JDark = DepthMap;
numpx = floor(imsize/1000);
JDarkVec = reshape(JDark,imsize,1);% size [w*h 1]
ImVec = reshape(I,imsize,3);% size [w*h 3]
[~, indices] = sort(JDarkVec);
size(indices)
indices = indices(imsize-numpx+1:end); % 取JDarkVec最大的前0.001的像素点

Id = zeros(size(I));
for i = 1:3
    Id(:,:,i) = DepthMap;%三通的depthmap
end
size(ImVec(indices(:), :))
A = mean(ImVec(indices(:), :));% 取depth前0.001的所有像素点之后分别对三个通道求均值
size(A)