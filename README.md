# Species-Detection-Of-Flowers-From-Images
Developing a framework for recognizing the species of a given flower. Implementation pipeline included Image Segmentation using 'Bi-level Co-Segmentation Method' followed by Image Classificatin/Recognition which includes feature extraction and then classification using SVM. </br>

Initial segmentation of the flowers is done using grabcut.py </br> 
Superpixels are obtained on the original images using slic.py </br>
Feature_Segmentation.m computes the feature vector on all the superpixels of the training images. </br>
Feature_Classification.m computes the feature vector on the testing images. </br>
main_segmentation.m consists of the training pipeline. </br>
main_classification.m consists of the testing pipeline. </br>
full_ovr.m cosists of the total one-vs-rest svm code which uses the functions ovrpredict.m and ovrtrain.m for classification. </br>
svm.m computhe the generic svm required in the testing pipeline which classifies pixels as eithe foreground or background. </br>
get_cv_ac.m gives the accuracy of the system. </br>
kernel_lda.m and kernel_pca.m are used for dimensionality reduction of the feature vectors.
