%% Prepare the data

% reading the data and assigning values

data=readtable('LCloanbook.xls');
data2=table2array(data);

Y = data2(:,1);
X=data2(:,2:end);
X_Red=data(:,{'annual_inc','delinq_2yrs','dti','emp_length','holiday_trips','int_rate','loan_amnt','revol_util','term','verification_status_Not_Verified' });
X_Red=table2array(X_Red);

% Splitting and permutating the data

[COEFF,SCORE] = princomp(X);
[n,m]= size(data2);
%data2=data2(randperm(n),:);
n10=(n-6)/10;
K_folds=mat2cell(data2,[6123;repmat(n10,9,1)], 124);

% Creating the Training and valuation sets

indices=(1:10)';
training=cell(10,1);
validation=cell(10,1);
B=zeros(124,10); 
Ypred=cell(10,1);
accuracies = zeros(10,1);

%% Logistic regression (10-fold CV)

% Full Model

for i = 1:10
    idx=[i];
    idx_set=setdiff(indices,idx);
    training{i}=cell2mat(K_folds(idx_set));
    validation{i}=cell2mat(K_folds(idx));
    X = training{i}(:,2:end);
    Y = categorical(training{i}(:,1));
    B=mnrfit(X,Y);
    Ypred{i}=mnrval(B,validation{i}(:,2:end));
    %Ypred{i}=[ones(length(validation{i}(:,2:end)),1),validation{i}(:,2:end)] * B;
    Ypred{i}(:,3)=Ypred{i}(:,1)<0.5;
    N=length(validation{i}(:,1));
    accuracies(i)=sum(not(xor(Ypred{i}(:,3),validation{i}(:,1))))/N;
end

accuracy_Logistic = mean(accuracies);

% Reduced Model

% -------------- ANALOGICAL -------------------

%% 1KNN

% Full Model

mdl_KNN=fitcknn(X,Y,'NumNeighbors',1); % The original values of X and Y
CV_KNN=crossval(mdl_KNN);
loss_CVKNN=kfoldLoss(CV_KNN);
accuracy_1KNN_Full=1-loss_CVKNN;

% Reduced Model

mdl_KNN_Reduced=fitcknn(X_Red,Y,'NumNeighbors',1);
CV_KNN_Reduced=crossval(mdl_KNN_Reduced);
loss_CVKNN_Reduced=kfoldLoss(CV_KNN_Reduced);
accuracy_1KNN_Red=1-loss_CVKNN_Reduced;

%% Classification Tree

% Full Model

tree_Class=fitctree(X,Y,'CrossVal','on','MaxNumSplits',100);
loss_Tree_Full=kfoldLoss(tree_Class);
accuracy_Tree_Full=1-loss_Tree_Full;

% Reduced Model

tree_Class_Red=fitctree(X_Red,Y,'CrossVal','on','MaxNumSplits',100);
loss_Tree_Red=kfoldLoss(tree_Class_Red);
accuracy_Tree_Red=1-loss_Tree_Red;

%% LASSO with Logistic

for i = 1:10
    % training and validation construction
    idx=[i];
    idx_set=setdiff(indices,idx);
    training{i}=cell2mat(K_folds(idx_set));
    validation{i}=cell2mat(K_folds(idx));
    
    % X and Y
    X = training{i}(:,2:end);
    Y = training{i}(:,1);
    
    % model estimation
    [B, FitInfos]=lassoglm(X,Y,'binomial','DFmax',10);
    
    %Plot the cross-validated fits.
    %lassoPlot(B,FitInfos,'PlotType','CV');
    
    %Locate the Optimal Lambda
    %bestLambda = find(FitInfos.Lambda == FitInfos.LambdaMinMSE);
    %bestLambda=FitInfos.Lambda(bestLambda);
    
    B=B(:,1); % Corresponding to the optimal Lambda /Tuning Lambda/
    
    B=[FitInfos.Intercept(1);B]; %Adding the intercept
    
    Ypred{i}=mnrval(B,validation{i}(:,2:end));
    predictors{i}=B;
    
    Ypred{i}(:,3)=Ypred{i}(:,1)>0.5;
    
    % accuracies
    N=length(validation{i}(:,1));
    accuracies(i)=sum(not(xor(Ypred{i}(:,3),validation{i}(:,1))))/N;
end

accuracy_LASSO = mean(accuracies);    

%% ENSEMBLES

% AdaBoost

t = templateTree('MaxNumSplits',100);
Mdl = fitcensemble(X,Y,'Method','AdaBoostM1','Learners',t,'CrossVal','on');
kflc = kfoldLoss(Mdl,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

% Bagging

t = templateTree('MaxNumSplits',100);
Mdl = fitcensemble(X,Y,'Method','Bag','Learners',t,'CrossVal','on');
kflc = kfoldLoss(Mdl,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

%% Support Vector Machine (SVM)

%% LASSO with ENSEMBLE (AdaBoostM1)
for i = 1:10
    % training and validation construction
    idx=[i];
    idx_set=setdiff(indices,idx);
    training{i}=cell2mat(K_folds(idx_set));
    validation{i}=cell2mat(K_folds(idx));
    
    % X and Y
    X = training{i}(:,2:end);
    Y = training{i}(:,1);
    
    % model estimation
    [B, FitInfos]=lasso(X,Y,'DFmax',10);
    B=B(:,1);
    %B=[FitInfos.Intercept(1);B];
    k=find(B(:,1));
    t = templateTree('MaxNumSplits',100);
    Mdl = fitcensemble(X,Y,'Method','AdaBoostM1','Learners',t,'CrossVal','on');
    loss_Boost=kfoldLoss(Mdl);
    accuracies(i)=1-loss_Boost;
    
end

accuracy_LASSO_BOOST = mean(accuracies);    

%% LASSO with Classification Tree

for i = 1:10
    % training and validation construction
    idx=[i];
    idx_set=setdiff(indices,idx);
    training{i}=cell2mat(K_folds(idx_set));
    validation{i}=cell2mat(K_folds(idx));
    
    % X and Y
    X = training{i}(:,2:end);
    Y = training{i}(:,1);
    
    % model estimation
    [B, FitInfos]=lasso(X,Y,'DFmax',10);
    B=B(:,1);
    %B=[FitInfos.Intercept(1);B];
    k=find(B(:,1));
    tree_Class_Red=fitctree(X(:,k),Y,'CrossVal','on','MaxNumSplits',100);
    % 

    loss_Tree_Full=kfoldLoss(tree_Class_Red);
    accuracies(i)=1-loss_Tree_Full;
    
end

accuracy_LASSO_TREE = mean(accuracies);    

