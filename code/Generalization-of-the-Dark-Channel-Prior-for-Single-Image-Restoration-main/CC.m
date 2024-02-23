function sc = CC(I)
[h, w, ~] = size(I);

imsize = h*w;
Ivec = reshape(I,imsize,3);
avgIC = mean(Ivec, 1);
[~, Dl] = getColorCast(I);

if Dl <= 0
    sc = [1 1 1];
else
    sc = (max(max(avgIC), 0.1)./max(avgIC, 0.1)).^(1/max(sqrt(Dl), 1));
end



