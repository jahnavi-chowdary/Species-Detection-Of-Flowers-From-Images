# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import os
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
#print type(args['image'])
u=args['image']
aaa=u.split('.')
#print type(aaa[0][0])

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
 
# loop over the number of segments
#for numSegments in (100, 200, 200):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
image =  cv2.resize(image,(200,200))
segments = slic(image, 100, sigma = 4)
aaaaa=[0]*100				# number of pixels in each superpixel

for ii in segments:
	#print type(ii)
	for i in ii:
#		print type(i)
		aaaaa[i]=aaaaa[i]+1
a = np.asarray(segments)
#print aaaaa
sumi=0
for i in aaaaa:
	sumi=sumi+i

average=sumi/len(aaaaa)
#print sumi
#print average

i=0
jj=[]
while i < 100:
	yy=[]
	if aaaaa[i] < average:
		yy.append(1)
		yy.append(0)
	else :
		yy.append(0)
		yy.append(1)
	jj.append(yy)
	i=i+1
#print jj
hh=np.matrix(jj)
#print hh
''' hhh.csv  has the size of each superpixel'''
np.savetxt("hhh.csv",hh, delimiter=",")
#print aaaaa
os.rename("hhh.csv", "../new/size_"+aaa[0]+".csv")#for ii in segments:
#	print (ii)
a = np.asarray(segments)
print a
'''image.csv has superpixel belonging of each pixel '''
np.savetxt("image.csv", a, delimiter=",")

os.rename("image.csv", "../csv/"+aaa[0]+".csv")
	# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (100))


''' detect edge '''


corner=[0]*100              #5) point

i=199
j=0
while(j<200):
	corner[a[j][i]]=1
	j=j+1

i=199
j=0
while(j<200):
	corner[a[i][j]]=1
	j=j+1

i=0
j=0
while(j<200):
	corner[a[j][i]]=1
	j=j+1
i=0
j=0
while(j<200):
	corner[a[i][j]]=1
	j=j+1

#print corner

edge=np.matrix(corner)
#print edge
#print edge.shape

''' he.csv shows whether the pixel is at the edge or middle of the superpixel'''
np.savetxt("he.csv",edge, delimiter=",")
os.rename("he.csv", "../edge/edge_"+aaa[0]+".csv")#for ii in segments:


ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
#plt.axis("on")
 
# show the plots
#plt.show()
