
import os, struct
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from array import array as pyarray
#from numpy import append, array, int8, uint8, zeros as np
import numpy as np

def load_mnist(dataset="training", digits=range(10), path='C:\\Users\\Shashi\\Downloads\\Contracts\\UCB\\UCB Ext\\Python\\MNIST_data'):
    
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        #fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.mod')
        #fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.small')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx')
        fname_lbl = os.path.join(path, 't10k-labels.idx')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    #images = np.zeros((N, rows, cols), dtype=np.uint8)
    images = np.zeros((N, rows * cols), dtype=np.uint8)
    small_size = 2
    images_small = np.zeros((N, small_size), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        #images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]
	# mod
	if (small_size ==2) :
		# 2 elements:
		images_small[i] = np.array([images[i][9*28 +16],images[i][15*28 +9]])
	else :
		# 3 elements:
		images_small[i] = np.array([images[i][9*28 +16],images[i][20*28 +18],images[i][15*28 +9]])
        #images[i][9*28 +16] = 0 # 7 diff
        #images[i][15*28 +9] = 0 # 4 diff
        #images[i][20*28 +18] = 0 # straigth

    #return images, labels
    return images_small, labels

#from pylab import *
#from numpy import *
#import scipy.sparse as sparse
#import scipy.linalg as linalg

clabel_n = 4
clabel_p = 7

#images, labels = load_mnist('training', digits=[4,7],path=os.getcwd())
#X, T = load_mnist('training', digits=[4,7],path=os.getcwd())
X, T = load_mnist('training', digits=[clabel_n,clabel_p],path=os.getcwd())

# converting from NX28X28 array into NX784 array
#flatimages = list()
#for i in images:
#    flatimages.append(i.ravel())
#X = np.asarray(flatimages)

'''
print("Check shape of matrix", X.shape)
print("Check Mins and Max Values",np.amin(X),np.amax(X))
print("\nCheck training vector by plotting image \n")
for i in range(len(X)) :
	if (len(X[i]) == 28*28) :
		plt.imshow(X[i].reshape(28, 28),interpolation='None', cmap=plt.get_cmap('gray'))
		plt.title(str(T[i]))
	else :
		plt.imshow(X[i].reshape(1, len(X[i])),interpolation='None', cmap=plt.get_cmap('gray'))
		plt.title(str(T[i])+str(X[i]))
	plt.show()
'''

print X
print T

X_mean = np.mean(X,axis=0).astype(float) #Avoiding potential issues with int
print X_mean

Z = X - X_mean
print Z

C = np.cov(X, rowvar=False, ddof=1)
print C
C_z = np.cov(Z, rowvar=False, ddof=1)
print C_z
C_alt = (np.dot(Z.transpose(), Z))/(Z.shape[0]-1)
print Z.shape[0]
print C_alt

[W,V_t]=np.linalg.eigh(C);
print W
print V_t
# Adjust:
W=np.flipud(W);V_t=np.fliplr(V_t);

print W
print V_t

#print type(X)
# Verify:
dim = V_t.shape[1]
for i in range(dim) :
	test_v = V_t[:,i]
	#verif = np.round(np.dot(C,test_v)/(W[0]*test_v),3)
	#verif = np.dot(C,test_v)-(W[0]*test_v)
	verif = np.round(np.dot(C,test_v)-(W[i]*test_v),8)
	if (not np.all(verif==0)) :
		print np.all(verif==1)
		print "Vector",i,"eigen verification failed:"
		print verif
		exit(1)
	#print "Verification Passed"
	for j in range(dim) :
		if (i != j) :
			test_v2 = V_t[:,j]
			ort = np.round(np.dot(test_v,test_v2),9)
			if (ort != 0) :
				print "Vector",i,"and",j,"are not orthogonal:",ort
				exit(1)

print "Verification Passed"

del dim 

P = np.dot(Z,V_t)
#P = np.dot(Z,V_t)
print P
#print P[:,0]
#print P[:,1]

# This is how we determine which plot to do:
#print X.shape[1]
index_n = np.where(T==clabel_n)[0]
index_p = np.where(T==clabel_p)[0]
#print X[index_n]
#print T[index_n]
#print X[index_p]
#print T[index_p]
#exit()

graph = True
if X.shape[1] == 3 and graph :
	fig = plt.figure()
	#ax = fig.add_subplot(121, projection='3d')
	ax = fig.add_subplot(221, projection='3d')

	ax.scatter(X[index_n][:,0], X[index_n][:,1], X[index_n][:,2], color='r', marker='o')
	ax.scatter(X[index_p][:,0], X[index_p][:,1], X[index_p][:,2], color='b', marker='o')
	ax.scatter(X_mean[0],X_mean[1],X_mean[2],color='k', marker='s')
	ax.set_xlabel('7 pixel')
	ax.set_ylabel('Straight')
	ax.set_zlabel('4 pixel')
	ax.axis('equal')

	#ax2 = fig.add_subplot(122, projection='3d')
	ax2 = fig.add_subplot(222, projection='3d')
	ax2.scatter(Z[index_n][:,0], Z[index_n][:,1], Z[index_n][:,2], color='r', marker='o')
	ax2.scatter(Z[index_p][:,0], Z[index_p][:,1], Z[index_p][:,2], color='b', marker='o')
	for i in range(len(W)) :
		plt.plot([0,np.sqrt(W[i])*V_t[0,i]],[0,np.sqrt(W[i])*V_t[1,i]],[0,np.sqrt(W[i])*V_t[2,i]],'k-',linewidth=2.0)
	plt.axis('equal')

	ax2.set_xlabel('7 pixel')
	ax2.set_ylabel('Straight')
	ax2.set_zlabel('4 pixel')

	#ax3 = fig.add_subplot(122, projection='3d')
	ax3 = fig.add_subplot(223, projection='3d')
	ax3.scatter(P[index_n][:,0], P[index_n][:,1], P[index_n][:,2], color='r', marker='o')
	ax3.scatter(P[index_p][:,0], P[index_p][:,1], P[index_p][:,2], color='b', marker='o')
	plt.axis('equal')

	ax3.set_xlabel('V1')
	ax3.set_ylabel('V2')
	ax3.set_zlabel('V3')

	plt.show()


if X.shape[1] == 2 and graph :
	alpha_cloud = 0.1
	plt.subplot(2,2,1)
	plt.scatter(X[index_n][:,0],X[index_n][:,1],color='r',marker="o",alpha=alpha_cloud)
	plt.scatter(X[index_p][:,0],X[index_p][:,1],color='b',marker="o",alpha=alpha_cloud)
	plt.plot(X_mean[0],X_mean[1],"ks")
	plt.axis('equal')
	plt.xlabel('7 Pixel')
	plt.ylabel('4 Pixel')
	plt.grid()

	#f2 = plt.figure(2)
	plt.subplot(2,2,2)
	plt.scatter(Z[index_n][:,0],Z[index_n][:,1],color="r",marker="o",alpha=alpha_cloud)
	plt.scatter(Z[index_p][:,0],Z[index_p][:,1],color="b",marker="o",alpha=alpha_cloud)
	for i in range(len(W)) :
		plt.plot([0,np.sqrt(W[i])*V_t[0,i]],[0,np.sqrt(W[i])*V_t[1,i]],'k-',linewidth=2.0)
	plt.axis('equal')
	plt.grid()

	#f3 = plt.figure(3)
	plt.subplot(2,2,3)
	plt.scatter(P[index_n][:,0],P[index_n][:,1],color="r",marker="o",alpha=alpha_cloud)
	plt.scatter(P[index_p][:,0],P[index_p][:,1],color="b",marker="o",alpha=alpha_cloud)
	plt.axis('equal')
	plt.grid()

	plt.show()

'''
'''





