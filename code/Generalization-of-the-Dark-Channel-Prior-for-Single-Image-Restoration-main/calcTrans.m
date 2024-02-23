function T = calcTrans(I, BL, win)

[h, w, ~] = size(I);
BLmap = zeros(size(I));
for ind = 1:3
    BLmap(:,:,ind) = BL(ind) * ones(size(h, w));
end
DLmap = abs(BLmap-I);
maxDL = zeros(size(I));
DLmapNor=zeros(size(I));
for ind = 1:3
    Bm = max(BL(ind), 1-BL(ind));
    maxDL(:,:,ind) = ordfilt2(DLmap(:,:,ind)/Bm,win^2,ones(win,win),zeros(win,win),'symmetric');
    DLmapNor(:,:,ind) = DLmap(:,:,ind)/Bm;
end
maxDL1= max(DLmapNor, [], 3);
T = medfilt2(maxDL1, [win win], 'symmetric');

T(T < 0) = 0;
T(T > 1) = 1;
