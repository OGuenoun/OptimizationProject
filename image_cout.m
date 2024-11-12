function image_cout(nfig,w)
figure(nfig)
npoints=1000;
xsample=linspace(-12, 12, npoints);
ysample=linspace(-12, 12, npoints);
[X,Y]=meshgrid(xsample,ysample);
z_opt=w(1)+w(2)*X+w(3)*Y;
class_opt=1./(1+exp(-z_opt));
colormap('jet')
image(xsample,ysample,class_opt,'CDataMapping','scaled')
caxis([0 1])
colorbar




