function [a ] = plot_field( it,field,xyz,Ne,mata,imat )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

a = 1
its = zeros(4,4);
ite = zeros(4,4);
ite(1,:)=[1,2,5,4];
ite(2,:)=[2,3,6,5];
ite(3,:)=[4,5,8,7];
ite(4,:)=[5,6,9,8];
for i=1:Ne
    if(mata(i)==imat||imat==0)
    its(1,:)=[it(i,1),it(i,2),it(i,5),it(i,4)];
    its(2,:)=[it(i,2),it(i,3),it(i,6),it(i,5)];
    its(3,:)=[it(i,4),it(i,5),it(i,8),it(i,7)];
    its(4,:)=[it(i,5),it(i,6),it(i,9),it(i,8)];
    for j=1:4
        x = xyz(its(j,:),1);
        y = xyz(its(j,:),2);
%        c = field(i,ite(j,:));
         c = field(its(j,:));
        h=patch(x,y,c);
        set(h,'LineStyle','none','FaceColor','interp');
        hold on;
    end
    end
end
colorbar vertical;
hbar = colorbar;
set(hbar,'TickLabelInterpreter','latex');
colormap jet;
axis equal;
axis off;
hold off;

end

