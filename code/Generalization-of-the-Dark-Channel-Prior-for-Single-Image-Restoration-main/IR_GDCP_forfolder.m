clear all;close all;
Stretch = @(x) (x - min(x(:))).*(1/(max(x(:)) - min(x(:))));
%win = 15;
win = 15;
t0 = 0.2;
r0 = t0 * 1.5;
sc = 1;
root='..\..\example';
names = dir([root '\input\']);
namess = { names.name };
names = namess( 3 : length(namess) );
for name = names
    name = char(name)
    im = im2double(imread([root '\input\' name]));
    [~, width, ~] = size(im);
    I = im2double(im);
    s = CC(I);
    [DepthMap, GradMap] = GetDepth(I, win);
    A = atmLight(I, DepthMap);
    T = calcTrans(I, A, win);
    maxT = max(T(:));
    minT = min(T(:));
    T_pro  = ((T-minT)/(maxT-minT))*(maxT-t0) + t0;
    Jc = zeros(size(I));
    for ind = 1:3 
        Am = A(ind)/s(ind);
        Jc(:,:,ind) = Am+(I(:,:,ind)-Am)./max(T_pro, r0);
    end
    Jc(Jc < 0) = 0;
    Jc(Jc > 1) = 1;
    T3 = cat(3, T, T, T);
    imwrite(T_pro, [root, '\T_pro\', name]);
    imwrite(DepthMap, [root, '\depth\', name]);
    imwrite(GradMap, [root, '\grad\', name]);
end
