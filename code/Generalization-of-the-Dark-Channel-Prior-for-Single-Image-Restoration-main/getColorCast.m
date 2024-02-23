function  [D, Dl] = getColorCast(im)

[L,a,b] = getRGB2Lab(im(:,:,1),im(:,:,2),im(:,:,3));
var_ansa = var(a(:));
var_ansb = var(b(:));

var_sq = sqrt(var_ansa + var_ansb);
u = sqrt(mean(a(:))^2 + mean(b(:))^2);
D = u - var_sq;
Dl = D/var_sq;


%color_cast = (u > 1) && (Dl > 0.5);

end

