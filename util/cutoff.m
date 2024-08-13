function [A1,A2,B1,B2] = cutoff(original)
%cut off black edge of the registered image
%writen by :Bian Chang
%Date:2020/05/22
[m,n]=size(original);     
b=sum(original,2);        
b=b';              
z=find(b>=1000000);   
[mm, nn]=size(z);
A1 = z(1);
A2 = z(nn);    %top and bottom row num

a=sum(original);         
z=find(a>=845000);    
[mm, nn]=size(z);  
B1 = z(1);
B2 = z(nn);   %top and bottom col num

end