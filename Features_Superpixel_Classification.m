function [features_matrix] = Features_Superpixel_Classification(input,labels_sp,size_sp,edge_sp)

un = unique(labels_sp);
total_sp = size(un,1);

features_matrix = [];

% Need to run this loop for all the superpixels i.e from 1:100    
for i = 0:99  
    i;
    flag0 = 0;
    flag1 = 1;
    clear row;
    clear col;
    [row,col] = find(labels_sp==i);
    indices = [row, col];
    siz = size(indices,1);
                
    %Getting features of the Superpixel
    superpix_full = zeros(h,w,3); 
        
    for k = 1:siz
        temp_ind = indices(k,:);
        ind_i = temp_ind(1,1);
        ind_j = temp_ind(1,2);
        superpix_full(ind_i,ind_j,:) = input(ind_i,ind_j,:); %superpix_full is the full image with superpixel
        superpix = superpix_full;
    end
    
    bw = rgb2gray(superpix);
    STATS = regionprops(bw,'BoundingBox');
    varia = STATS.BoundingBox;
    if (round(varia(1,2))+varia(1,4)) > h
        hei_diff = h;
    else
        hei_diff = (round(varia(1,2))+varia(1,4));
    end
    
    if (round(varia(1,1))+varia(1,3)) > w
        wid_diff = w;
    else
        wid_diff = (round(varia(1,1))+varia(1,3));
    end
    
    super = superpix_full(round(varia(1,2)):hei_diff , round(varia(1,1)):wid_diff , :);
    superpix1 = uint8(super);

    % (1) 200 Dimensional Lab Features
            
    Ilab = superpix1;
    Ilab = rgb2lab(Ilab);
            
    Ilab_l = Ilab(:,:,1);
    Ilab_a = Ilab(:,:,2);
    Ilab_b = Ilab(:,:,3);
            
    Ilab_l = Ilab_l(:);
    Ilab_a = Ilab_a(:);
    Ilab_b = Ilab_b(:);
            
    descr = [Ilab_l,Ilab_a,Ilab_b];
    noofframes = size(descr,1);
    size(descr)
    if noofframes < 200
        [C1, A1] = vl_kmeans(im2single(descr'), noofframes);
        diff = 200 - noofframes;
        zero = zeros(1,diff);
    else
        [C1, A1] = vl_kmeans(im2single(descr'), 200);
    end
    C1 = C1';
    for j = 1:size(descr,1)
        Xl = descr(j,:);
        Dl = pdist2(Xl,C1);
        [minvalsl, minindsl] = min(Dl, [], 2);
        hisl(j,1) = minindsl;
    end
     if noofframes < 200
        counts1 = hist(hisl,noofframes);
        des1 = [counts1,zero];
    else
        counts1 = hist(hisl,200);
        des1 = counts1;
    end
    size(des1);

    % (2) 800 Dimensional SIFT Features
            
    I = superpix1 ;
    clear FRAMES1;
    clear DESCRS1;
    [FRAMES1,DESCRS1] = vl_dsift(rgb2gray(im2single(I)));
    FRAMES1 = FRAMES1';
    DESCRS1 = DESCRS1';
    noofframes = size(DESCRS1,1);
    if noofframes < 800
        [C, A] = vl_kmeans(im2single(DESCRS1'), noofframes);
        diff = 800 - noofframes;
        zero = zeros(1,diff);
    else
        [C, A] = vl_kmeans(im2single(DESCRS1'), 800);
    end
    C = C';
    for j = 1:size(DESCRS1,1)
        X = DESCRS1(j,:);
        D = pdist2(X,C);
        [minvals, mininds] = min(D, [], 2);
        his(j,1) = mininds;            
    end
    if noofframes < 800
        counts = hist(his,noofframes);
        des2 = [counts,zero];
    else
        counts = hist(his,800);
        des2 = counts;
    end
    size(des2);
           
    % (3) Size - 0/1
    des3 = size_sp(i+1,:);        
            
    % (4) 36 D Location
    bin1 = superpix_full;
    bin1 = im2bw(bin1);
    bin1 = imresize(bin1,[6 6]);
    des4 = [bin1(:)]';
         
    % (5) 36 D Shape
    bin2 = superpix1;
    bin2 = im2bw(bin2);
    bin2 = imresize(bin2,[6 6]);
    des5 = [bin2(:)]';
                   
    % (6) Centre or Boundary
    des6 = edge_sp(i+1,:);        
            
    % Final Feature Vector
    des = [des1,des2,des3,des4,des5,des6];
    size(des);
    
    features_matrix = [features_matrix;des];
    
    clear superpix1;  
    clear superpix_full;
    clear super;
    
end
