% Hebbian Learning.
% This script implements Hebbian learning.
% Training_E.csv, Test_I_E.csv, Test_II_E.csv, and Test_III_E.csv are the
% csv files containing the training and test sets respectively. The same
% applies for A.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
Train_set=csvread('Training_E.csv');
[m,n]=size(Train_set);

w=zeros(21,1);
for i=1:n
    w=w+Train_set(1:21,i)*Train_set(22,i);
end

%Predictions 

test_1=csvread('Test_I_E.csv');
T=test_1(1:21,:)'; %matrix containing test instances
pred=arrayfun(@activation,(T*w)); % multiply T by w to get 
%the predictions for each instance.

test_2=csvread('Test_II_E.csv');
T=test_2(1:21,:)';
pred=arrayfun(@activation,(T*w));

test_3=csvread('Test_III_E.csv');
T=test_3(1:21,:)';
pred=arrayfun(@activation,(T*w));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


