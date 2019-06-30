% This script implements the perceptron learning algorithm.
% The user should set alpha, tol (tolerance) and maxepoch.
% All the plots required in the project are programmed.
% Training_E.csv, Test_I_E.csv, Test_II_E.csv, and Test_III_E.csv are the
% csv files containing the training and test sets respectively. The same
% applies for A.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
Train_set=csvread('Training_E.csv');
[m,n]=size(Train_set);
alpha=0.01;
w=zeros(21,1);
mx_wt=0;
tol=0.001;
maxepoch=500;
wc_v=[];
W=[];
miss=0;
for j=1:maxepoch
    
for i=1:n
    y_net=w'*Train_set(1:21,i);
    y=activation(y_net); 
    
    if y~=Train_set(22,i)
        miss=miss+1;
        deltaw=alpha*Train_set(22,i)*Train_set(1:21,i);
        w=w+deltaw;
        mx_wp=max(abs(deltaw));
        
        if mx_wp>mx_wt
            mx_wt=mx_wp;
        end    
    end
    ind=[1:5,21];
    W=[W,w(ind)];        
end

wc_v=[wc_v,mx_wt];

if mx_wt<tol
    break
end
    
end


%plot to show the maximum weight change (absolute value) in each epoch.
plot(wc_v);
grid on
title('Maximum Weight change across epochs')
xlabel('Epoch')
ylabel('Max Weight Change')


%plots to show the evolution of weights.
figure(1);
subplot(2,1,1);
plot(W(1,:));
grid on
axis([0 500 0 3])
title('Evolution of W1 across epochs')
xlabel('Epoch')
ylabel('Weight')

subplot(2,1,2);
plot(W(2,:));
grid on
axis([0 500 -1.5 3.5])
title('Evolution of W2 across epochs')
xlabel('Epoch')
ylabel('Weight')

figure(2);
subplot(2,1,1);
plot(W(3,:));
grid on
axis([0 500 -4.5 1.5])
title('Evolution of W3 across epochs')
xlabel('Epoch')
ylabel('Weight')

subplot(2,1,2);
plot(W(4,:));
grid on
axis([0 500 -1.5 4.5])
title('Evolution of W4 across epochs')
xlabel('Epoch')
ylabel('Weight')

figure(3);
subplot(2,1,1);
plot(W(5,:));
grid on
axis([0 500 0 6.5])
title('Evolution of W5 across epochs')
xlabel('Epoch')
ylabel('Weight')

subplot(2,1,2);
plot(W(6,:));
grid on
axis([0 500 -6.5 0])
title('Evolution of bias term across epochs')
xlabel('Epoch')
ylabel('Weight')

%Predict using the tained network.

test_1=csvread('Test_I_E.csv');
T=test_1(1:21,:)'; %matrix containing test instances
pred=arrayfun(@activation,(T*w)); % multiply T by w 
%to get the predictions for each instance.

test_2=csvread('Test_II_E.csv');
T=test_2(1:21,:)'; %matrix containing test instances
pred=arrayfun(@activation,(T*w)); % multiply T by w 
%to get the predictions for each instance.

test_3=csvread('Test_III_E.csv');
T=test_3(1:21,:)'; %matrix containing test instances
pred=arrayfun(@activation,(T*w)); % multiply T by w 
%to get the predictions for each instance.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
