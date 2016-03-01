close all;
clc;
% clear;

dir_input = dir('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/orig_input/*.jpg');
dir_seg = dir('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/segmented/*.jpg');
dir_suppix = dir('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/csv/*.csv');
dir_des4 = dir('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/size/*.csv');
dir_des5 = dir('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/edge/*.csv');

feature_vector = [];
full_labels = [];

tic
for k = 1:length(dir_input)
    
    % INPUT IMAGE
    fname1 = dir_input(k).name;
    [path,name,ext] = fileparts(fname1); % separate out base name of file
    images.(name) = imread(fullfile('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/orig_input', fname1));
    input = images.(name);
    input = imresize(input,[200 200]);
    %figure; imshow(im);
    
    %SEGMENTED IMAGE
    fname2 = dir_seg(k).name;
    [path,name,ext] = fileparts(fname2); % separate out base name of file
    images.(name) = imread(fullfile('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/segmented', fname2));
    seg = images.(name);
    seg = imresize(seg,[200 200]);
    
    %LABELS OF THE SUPERPIXELS OF THE SEGMENTED IMAGE
    fname3 = dir_suppix(k).name;
    [path,name,ext] = fileparts(fname3); % separate out base name of file
    labels_sp = csvread(fullfile('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/csv', fname3));
    
    %SIZE OF SUPERPIXEL (10/01)
    fname4 = dir_des4(k).name;
    [path,name,ext] = fileparts(fname4); % separate out base name of file
    size_sp = csvread(fullfile('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/size', fname4));
    
    %PERIPHERY/CENTRE SUPERPIXEL (00/01)
    fname5 = dir_des5(k).name;
    [path,name,ext] = fileparts(fname5); % separate out base name of file
    edge_sp = csvread(fullfile('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/edge', fname5));
    edge_sp = edge_sp' ;
    
    tic;
    [features,labels] = Features_Segmentation(input,seg,labels_sp,size_sp,edge_sp);
    labels = labels';
    
    feature_vector = [feature_vector;features];
    full_labels = [full_labels;labels];
    toc;
    
end
toc

% Using Kernel PCA 
feature_vector = kernel_pca(feature_vector,500);

% Using Kernel LDA 
[eigvector, eigvalue, elapse] = kernel_lda(0,train_lbl,feature_vector_train);


[final_labels_sp,model] = svm(feature_vector,full_labels);

% Reestimating the labels of the pixels based on the obtained model
for k = 1:length(dir_input)
    
    %LABELS OF THE SUPERPIXELS OF THE SEGMENTED IMAGE
    fname3 = dir_suppix(k).name;
    [path,name,ext] = fileparts(fname3); % separate out base name of file
    labels_sp = csvread(fullfile('C:/Users/Jahnavi Chowdary/Desktop/IIIT/3rd Year/SMAI/Project/codes_smai_project/Final_new/csv', fname3));
    labels_from_svm = final_labels_sp(((100 * (k-1)) + 1):(100 * k),:);
    
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
    full_file_name = fullfile('C:\Users\Jahnavi Chowdary\Desktop\IIIT\3rd Year\SMAI\Project\codes_smai_project\Final_new\final_segmented',file_name);  
    
    imwrite(final_segmented_image,full_file_name);
    
end





