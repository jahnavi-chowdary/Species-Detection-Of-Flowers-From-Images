close all;
clc;

% --------------- TRAINING PIPELINE --------------- %

dir_seg = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\seg_train\*.jpg');
len = length(dir_seg);
no_of_classes = len./15;

full_labels_set = csvread('Labels.csv');

feature_vector_train = [];
train_label = zeros(len,1);

% Get the feature vectors of all the training samples
for k = 1:len
    
    %SEGMENTED IMAGE
    fname2 = dir_seg(k).name;
    [path,name,ext] = fileparts(fname2); % separate out base name of file
    images.(name) = imread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\seg_train', fname2));
    seg = images.(name);
    seg = imresize(seg,[200 200]);
    
    [token, remain] = strtok(fname2,'_');
    [token, remain] = strtok(remain,'_');
    [token, remain] = strtok(remain,'_');
    [token, remain] = strtok(token,'.'); % Number of the image is in token
    token = str2num(token);
    
    train_label(k,1) = full_labels_set(token,1);
    
    [des_lab1 des_sift1] = Features_Classification(seg);
    des_lab2 = vl_homkermap(des_lab1, 1) ;
    des_sift2 = vl_homkermap(des_sift1, 1) ;
    
    features_train = [des_lab2;des_sift2];
    
    feature_vector_train = [feature_vector_train;features_train];
    size(feature_vector_train);

end

% Getting the Generic SVM

dir_input = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\orig_input_train\*.jpg');
dir_seg = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\seg_train\*.jpg');
dir_suppix = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\csv_train\*.csv');
dir_des4 = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\size_train\*.csv');
dir_des5 = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\edge_train\*.csv');

feature_vector_superpixel = [];
full_labels_superpixel = [];

tic
for k = 1:length(dir_input)
    
    % INPUT IMAGE
    fname1 = dir_input(k).name;
    [path,name,ext] = fileparts(fname1); % separate out base name of file
    images.(name) = imread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\orig_input_train', fname1));
    input = images.(name);
    input = imresize(input,[200 200]);
    
    %SEGMENTED IMAGE
    fname2 = dir_seg(k).name;
    [path,name,ext] = fileparts(fname2); % separate out base name of file
    images.(name) = imread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\seg_train', fname2));
    seg = images.(name);
    seg = imresize(seg,[200 200]);
    
    %LABELS OF THE SUPERPIXELS OF THE SEGMENTED IMAGE
    fname3 = dir_suppix(k).name;
    [path,name,ext] = fileparts(fname3); % separate out base name of file
    labels_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\csv_train', fname3));
    
    %SIZE OF SUPERPIXEL (10/01)
    fname4 = dir_des4(k).name;
    [path,name,ext] = fileparts(fname4); % separate out base name of file
    size_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\size_train', fname4));
    
    %PERIPHERY/CENTRE SUPERPIXEL (00/01)
    fname5 = dir_des5(k).name;
    [path,name,ext] = fileparts(fname5); % separate out base name of file
    edge_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Train_Dataset\edge_train', fname5));
    edge_sp = edge_sp' ;
    
    tic;
    [features_superpixel,labels] = Features_Segmentation(input,seg,labels_sp,size_sp,edge_sp);
    labels = labels';
    
    feature_vector_superpixel = [feature_vector_superpixel;features_superpixel];
    full_labels_superpixel = [full_labels_superpixel;labels];
    toc;
    
end
toc


% Using Kernel PCA 
feature_vector_train = kernel_pca(feature_vector_train,30000);

% Using Kernel LDA
[eigvector, eigvalue, elapse] = kernel_lda(0,train_lbl,feature_vector_train);


[final_labels_sp,generic_model] = svm(feature_vector_superpixel,full_labels_superpixel);

% Getting the OVR SVM Classifier Model

ovr_model = full_ovr(train_labels,feature_vector_train);



% --------------- TESTING PIPELINE --------------- %

dir_input = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\orig_input_test\*.jpg');
dir_suppix = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\csv_test\*.csv');
dir_des4 = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\size_test\*.csv');
dir_des5 = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\edge_test\*.csv');

feature_vector_superpixel_test = [];

% Get the feature vectors of all the training samples
for k = 1:length(dir_input)
    
    % INPUT IMAGE
    fname1 = dir_input(k).name;
    [path,name,ext] = fileparts(fname1); % separate out base name of file
    images.(name) = imread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\orig_input_test', fname1));
    input = images.(name);
    input = imresize(input,[200 200]);
    
    %LABELS OF THE SUPERPIXELS OF THE SEGMENTED IMAGE
    fname3 = dir_suppix(k).name;
    [path,name,ext] = fileparts(fname3); % separate out base name of file
    labels_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\csv_test', fname3));
    
    %SIZE OF SUPERPIXEL (10/01)
    fname4 = dir_des4(k).name;
    [path,name,ext] = fileparts(fname4); % separate out base name of file
    size_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\size_test', fname4));
    
    %PERIPHERY/CENTRE SUPERPIXEL (00/01)
    fname5 = dir_des5(k).name;
    [path,name,ext] = fileparts(fname5); % separate out base name of file
    edge_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\edge_test', fname5));
    edge_sp = edge_sp' ;
    
    [features_superpixel_test] = Features_Superpixel_Classification(input,labels_sp,size_sp,edge_sp);
    
    feature_vector_superpixel_test = [feature_vector_superpixel_test;features_superpixel_test];   
    
    [predict_lbl_test, accuracy, dec_val] = svmpredict(trn_lbl, trn_ftr, generic_model);
    
    %LABELS OF THE SUPERPIXELS OF THE SEGMENTED IMAGE
    fname3 = dir_suppix(k).name;
    [path,name,ext] = fileparts(fname3); % separate out base name of file
    labels_sp = csvread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\csv_test', fname3));
    labels_from_svm = predict_lbl_test;
    
    for i = 0:99  % Need to run this loop for all the superpixels i.e from 1:300
        i;
        flag0 = 0;
        flag1 = 1;
        clear row;
        clear col;
        [row,col] = find(labels_sp==i);
        indices = [row, col];
        siz = size(indices,1);
        val_of_pix = labels_from_svm(i+1,:);
        for k = 1:siz
            
            % Getting the label of the Pixels
            temp_ind = indices(k,:);
            ind_i = temp_ind(1,1);
            ind_j = temp_ind(1,2);
            final_segmented_image_binary(ind_i,ind_j) = val_of_pix; 
            
        end
    end
    
    for i = 1:size(input,1)
        for j = 1:size(input,2)
            if final_segmented_image_binary == 1
                final_segmented_image(i,j,:) = input(i,j,:);
            end
        end
    end
    
    % GMM + Graph Cut
    [row_0,col_0] = find(final_segmented_image_binary==0);
    indices_0 = [row_0, col_0];
    siz_0 = size(indices_0,1);
    
    [row_1,col_1] = find(final_segmented_image_binary==1);
    indices_1 = [row_1, col_1];
    siz_1 = size(indices_1,1);
    
    feature_vector_image = feature_vector(((100 * (k-1)) + 1):(100 * k),:);
    feature_vector_0 = feature_vector_image(indices_0);
    feature_vector_1 = feature_vector_image(indices_1);
        
    % GMM for pixels in BG
    ftr = feature_vector_0;
    r = randi(size(ftr,1),[1 200000]);
    ftr = ftr(r,:);
    for k1 = 1:5
        k1
        obj{k1} = gmdistribution.fit(ftr,k1,'Start','randSample','Replicates',5,'CovType','full','SharedCov',false,'Regularize',.001,'Options',options);
        AIC(k1)= obj{k1}.AIC;
    end
    [minAIC,numComponents] = min(AIC);
    %numComponents
    gmm{i} = obj{numComponents};
    %gmm_0 = obj;
    
    % GMM for pixels in FG
    ftr = feature_vector_1;
    r = randi(size(ftr,1),[1 200000]);
    ftr = ftr(r,:);
    for k1 = 1:5
        k1
        obj{k1} = gmdistribution.fit(ftr,k1,'Start','randSample','Replicates',5,'CovType','full','SharedCov',false,'Regularize',.001,'Options',options);
        AIC(k1)= obj{k1}.AIC;
    end
    [minAIC,numComponents] = min(AIC);
    %numComponents
    gmm_1 = obj{numComponents};
    %gmm_1 = obj;
    
    all_segmented_images{k} = final_segmented_image;

    file_name = dir_seg(k).name;
    full_file_name = fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\seg_test',file_name);  
    
    imwrite(final_segmented_image,full_file_name);
   
end

dir_seg = dir('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\seg_test\*.jpg');
len = length(dir_seg);
no_of_classes = len./65;

feature_vector_test = [];
test_label = zeros(len,1);

% Get the feature vectors of all the testing samples
for k = 1:len
    
    %SEGMENTED IMAGE
    fname2 = dir_seg(k).name;
    [path,name,ext] = fileparts(fname2); % separate out base name of file
    images.(name) = imread(fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\Test_Dataset\seg_test', fname2));
    seg = images.(name);
    
    [token, remain] = strtok(fname2,'_');
    [token, remain] = strtok(remain,'_');
    [token, remain] = strtok(token,'.'); % Number of the image is in token
    token = str2num(token);
    
    test_label(k,1) = full_labels_set(token,1);
    
    [des_lab1 des_sift1] = Features_Classification(seg);
    des_lab2 = vl_homkermap(des_lab1, 1) ;
    des_sift2 = vl_homkermap(des_sift1, 1) ;
    
    features_test = [des_lab2;des_sift2];
    
    feature_vector_test = [feature_vector_test;features_test];
    size(feature_vector_test);
    
end

% Using Kernel PCA 
feature_vector_test = kernel_pca(feature_vector_test,30000);

% Using Kernel LDA
[eigvector, eigvalue, elapse] = kernel_lda(0,test_lbl,feature_vector_test);

% Predicting the class using the obtained OVR SVM Classifier Model

% % #######################
% % Classify samples using OVR model
% % #######################

[predict_label, accuracy, prob_values] = ovrpredict(test_Label, feature_vector_test, ovr_model);
fprintf('Accuracy = %g%%\n', accuracy * 100);
