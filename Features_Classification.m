function [des1 des2] = Features_Classification(seg)

% Binary image of segmented image -> binary
h = size(seg,1);
w = size(seg,2);
p = zeros(h,w);
for i = 1:h
    for j = 1:w
        if seg(i,j) == 0
            p(i,j) = 0;
        else
            p(i,j) = 1;
        end            
    end
end
SE = strel('disk',4);
p = imopen(p,SE);
binary = p;  

% Feature Extraction

% (1) 800 Lab Color Descriptors

lab_seg = rgb2lab(seg);

[row,col] = find(binary==1);
indices = [row, col];
siz = size(indices,1);

descr = [];

for k = 1:siz
    
    temp_ind = indices(k,:);
    ind_i = temp_ind(1,1);
    ind_j = temp_ind(1,2);
    
    Ilab = lab_seg;
            
    Ilab_l = Ilab(ind_i,ind_j,1);
    Ilab_a = Ilab(ind_i,ind_j,2);
    Ilab_b = Ilab(ind_i,ind_j,3);
            
    descr_pix = [Ilab_l,Ilab_a,Ilab_b];
    descr = [descr;descr_pix];
        
end

size(descr)
[Cl, Al] = vl_kmeans(im2single(descr'), 800);
Cl = Cl';
for j = 1:size(descr,1)
    Xl = descr(j,:);
    Dl = pdist2(Xl,Cl);
    [minvalsl, minindsl] = min(Dl, [], 2);
    hisl(j,1) = minindsl;
end
countsl = hist(hisl,800);
des1 = countsl;
size(des1);

% (2) 8000 x 3 SIFT Descriptors

% Dense SIFT Features

I = seg ;
[frames, descrs]=vl_phow(im2single(I)) ;
[FRAMES1,DESCRS1] = vl_dsift(rgb2gray(im2single(I)));
FRAMES1 = FRAMES1';
DESCRS1 = DESCRS1';
noofframes = size(DESCRS1,1);
if noofframes < 8000
    [C, A] = vl_kmeans(im2single(DESCRS1'), noofframes);
    diff = 8000 - noofframes;
    zero = zeros(1,diff);
else
    [C, A] = vl_kmeans(im2single(DESCRS1'), 8000);
end
C = C';
for j = 1:size(DESCRS1,1)
    X = DESCRS1(j,:);
    D = pdist2(X,C);
    [minvals, mininds] = min(D, [], 2);
    his(j,1) = mininds;            
end
if noofframes < 8000
    counts = hist(his,noofframes);
    des2_1 = [counts,zero];
else
    counts = hist(his,8000);
    des2_1 = counts;
end
size(des2_1);


% Interest points SIFT features

I = seg ;
I = single(rgb2gray(I)) ;
[FRAMES1,DESCRS1] = vl_sift(I);
FRAMES1 = FRAMES1';
DESCRS1 = DESCRS1';
noofframes = size(DESCRS1,1);
if noofframes < 8000
    [C, A] = vl_kmeans(im2single(DESCRS1'), noofframes);
    diff = 8000 - noofframes;
    zero = zeros(1,diff);
else
    [C, A] = vl_kmeans(im2single(DESCRS1'), 8000);
end
C = C';
for j = 1:size(DESCRS1,1)
    X = DESCRS1(j,:);
    D = pdist2(X,C);
    [minvals, mininds] = min(D, [], 2);
    his(j,1) = mininds;            
end

if noofframes < 8000
    counts = hist(his,noofframes);
    des2_2 = [counts,zero];
else
    counts = hist(his,8000);
    des2_2 = counts;
end
size(des2_2);

% Edge SIFT features

I = seg ;
I1 = im2bw(I);
I1 = edge(I1,'sobel');
I = single(rgb2gray(I)) ;

[row,col] = find(I1 == 1);
indices = [row, col];
siz = size(indices,1);

FRAMES1 = [];
DESCRS1 = [];

for k = 1:siz
    temp_ind = indices(k,:);
    ind_i = temp_ind(1,1);
    ind_j = temp_ind(1,2);
    
    fc = [ind_i;ind_j;10;0] ;
    [f,d] = vl_sift(I,'frames',fc,'orientations') ;
    
    FRAMES1 = [FRAMES1,f];
    DESCRS1 = [DESCRS1,d];
end

FRAMES1 = FRAMES1';
DESCRS1 = DESCRS1';

noofframes = size(DESCRS1,1);
if noofframes < 8000
    [C, A] = vl_kmeans(im2single(DESCRS1'), noofframes);
    diff = 8000 - noofframes;
    zero = zeros(1,diff);
else
    [C, A] = vl_kmeans(im2single(DESCRS1'), 8000);
end
C = C';
for j = 1:size(DESCRS1,1)
    X = DESCRS1(j,:);
    D = pdist2(X,C);
    [minvals, mininds] = min(D, [], 2);
    his(j,1) = mininds;            
end
if noofframes < 8000
    counts = hist(his,noofframes);
    des2_3 = [counts,zero];
else
    counts = hist(his,8000);
    des2_3 = counts;
end
size(des2_3);

des2 = [des2_1,des2_2,des2_3];



