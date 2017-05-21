
import os, struct
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from array import array as pyarray
#from numpy import append, array, int8, uint8, zeros as np
import numpy as np
import xlsxwriter

def load_mnist(dataset="training", digits=range(10), path='C:\\Users\\Shashi\\Downloads\\Contracts\\UCB\\UCB Ext\\Python\\MNIST_data'):
    
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        #fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        #fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.mod')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.small')
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
    small_size = 3
    images_small = np.zeros((N, small_size), dtype=np.uint8)
    #labels = np.zeros((N, 1), dtype=np.int8)
    #labels = np.zeros((N, 1), dtype=np.int8)
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

def bin_index(x, nb, xmin, xmax) :
	"Returns the bin number given the x"
	return	round(float((nb-1)*(x-xmin))/(xmax-xmin))

def bin_position(query, shape, xmin, xmax) :
	"Returns a tuple whith the position of the querry in N dim"
	return tuple(
	  int(bin_index(query[i],shape[i],xmin[i],xmax[i]))
	  for i in range(len(shape)) )

def histogram_counts(query, histograms, xmin, xmax) :
	"Returns the count of items from each histogram, they have to have same shape"
	coor = bin_position(query, histograms[0].shape,xmin, xmax)
	cnt = [ h[coor] for h in histograms ]
	return cnt

def histogram_clasifier(query, histograms, labels, xmin, xmax) :
	counts = np.array(histogram_counts(query, histograms, xmin, xmax))
	#print "From clasifier count",counts
	#counts = np.array([2,0])
	max_cnt = max(counts)
	#print max_cnt
	idxs = np.where(counts==max_cnt)[0]
	#print idxs
	#print len(idxs)
	if (len(idxs) == 1) :
		winning_label = labels[np.asscalar(idxs)]
	else :
		winning_label = "undetermined"
	if (max_cnt==0) :
		winning_prob = float('NaN')
	else :
		winning_prob = float(counts[idxs[0]])/np.sum(counts)
	return winning_label, winning_prob


def bin_center(n, nb, xmin, xmax) :
	"Returns the center of the bin range given the bin number"
	# There is no round to allow this to be numpy array
	return	n*(xmax-xmin)/float(nb-1) + xmin

def bin_bottom(n, nb, xmin, xmax) :
	"Returns the center of the bin range given the bin number"
	# There is no round to allow this to be numpy array
	return	(2*n-1)*(xmax-xmin)/float(2*(nb-1)) + xmin

def norm_pdf(x, mu, cov) :
	d = len(mu)
	#print d
	xcenter = np.matrix(x-mu)
	#print xcenter
	covinv = np.matrix(np.linalg.inv(cov))
	#print covinv
	exparg = -0.5 * np.asscalar((xcenter * covinv) * xcenter.transpose())
	#print exparg
	det = (np.linalg.det(cov)) 
	#print math.sqrt(abs(det))
	#print (math.sqrt(2*math.pi))**d
	div = ((math.sqrt(2*math.pi))**d) * math.sqrt(abs(det))
	#print div
	# det =-9.25185853854e-16 
	pdf =  ( math.exp(exparg) / div )
	#print pdf
	return pdf

def plot_histogram(neg_hist, pos_hist,row_max,row_min,col_max,col_min):
	print "Building bar graph"
	row_bins, col_bins = neg_hist.shape
	print "row_bins", row_bins
	print "col_bins", col_bins

	row_width = float(row_max-row_min)/(row_bins-1)
	col_width = float(col_max-col_min)/(col_bins-1)
	print "row_width",row_width
	print "col_width",col_width

	m_rows,m_cols = np.nonzero(neg_hist)
	m_val = neg_hist[m_rows,m_cols]
	f_rows,f_cols = np.nonzero(pos_hist)
	f_val = pos_hist[f_rows,f_cols]

	fig = plt.figure()
	ax1 = fig.add_subplot(111,projection='3d')
	ax1.bar3d(
		(bin_bottom(f_rows,row_bins,row_min,row_max)),
		(bin_bottom(f_cols,col_bins,col_min,col_max)),
		np.zeros(len(f_val),int),
		row_width,
		col_width,
		f_val,
		color='b',
		alpha=0.5)

	ax1.bar3d(
		(bin_bottom(m_rows,row_bins,row_min,row_max)),
		(bin_bottom(m_cols,col_bins,col_min,col_max)),
		np.zeros(len(m_val),int),
		row_width,
		col_width,
		m_val,
		color='r',
		alpha=0.5)

	ax1.set_xlabel('Row')
	ax1.set_ylabel('Col')

	plt.show()




np.random.seed(1)

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

print "Data set"
print X
print "Labels"
print T

X_mean = np.mean(X,axis=0).astype(float) #Avoiding potential issues with int
print "Mean vector"
print X_mean

Z = X - X_mean
print "Z matrix"
print Z

C = np.cov(X, rowvar=False, ddof=1)
print "Covariance matrix"
print C
print "Covariance matrix calculated from Z"
C_z = np.cov(Z, rowvar=False, ddof=1)
print C_z
C_alt = (np.dot(Z.transpose(), Z))/(Z.shape[0]-1)
#print Z.shape[0]
print "Covariance matrix calculated from ZxZ"
print C_alt

[W,V_t]=np.linalg.eigh(C);
print "Original eigen values"
print W
print "Original eigen vectors"
print V_t
# Adjust:
W=np.flipud(W);V_t=np.fliplr(V_t);

print "Sorted eigen values"
print W
print "Sorted eigen vectors"
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

print "Eigen Verification Passed"

del dim 

P = np.dot(Z,V_t)
print "Eigen adjusted data"
print P

P_red = P[:,0:2]
print "First 2 Eigen adjusted data"
print P_red

# This is how we determine which plot to do:
index_n = np.where(T==clabel_n)[0]
index_p = np.where(T==clabel_p)[0]

P_red_n = P_red[index_n]
P_red_p = P_red[index_p]

total_n = len(index_n)
total_p = len(index_p)

print "Negative features:"
print P_red_n
print "Total samples of N (",clabel_n,"):",total_n
print "Positive features:"
print P_red_p
print "Total samples of P (",clabel_p,"):",total_p

graph = False
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

	plt.subplot(2,2,4)
	alpha_cloud = 0.1
	plt.scatter(P_red[index_n][:,0],P_red[index_n][:,1],color="r",marker="o",alpha=alpha_cloud)
	plt.scatter(P_red[index_p][:,0],P_red[index_p][:,1],color="b",marker="o",alpha=alpha_cloud)
	plt.axis('equal')
	plt.grid()


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

# Range verification:
'''
X= np.array(
[[  2, 255, 253],
 [  3,  20, 252],
 [222,  30,   0],
 [ 49, 252, 252]]
)
'''
#X_max = np.amax(X,axis=0)
#X_min = np.amin(X,axis=0)
X_diff = (np.amax(X,axis=0)-np.amin(X,axis=0)).astype(float)
max_range = np.sqrt(np.dot(X_diff,X_diff))
#print X
#print X_min
#print X_max
#print X_diff
print "Expecting a max range of",max_range # 409.4313

P_red_max = np.amax(P_red,axis=0)
P_red_min = np.amin(P_red,axis=0)
P_red_diff = P_red_max-P_red_min
#print P_red
print "Max:",P_red_max
print "Min:",P_red_min
print "Diff",P_red_diff

if (not np.all(P_red_diff<max_range)) :
	print "Range verification failed, expecting a maximun range of", \
	       max_range, "and this is the array of ranges"
	print P_red_diff

'''
# Scatter for homework:
alpha_cloud = 0.1
plt.scatter(P_red[index_n][:,0],P_red[index_n][:,1],color="r",marker="o",alpha=alpha_cloud)
plt.scatter(P_red[index_p][:,0],P_red[index_p][:,1],color="b",marker="o",alpha=alpha_cloud)
plt.axis('equal')
plt.grid()

plt.show()
'''

# Classifier related:
P_red_mean_n = np.mean(P_red_n,axis=0).astype(float)
P_red_mean_p = np.mean(P_red_p,axis=0).astype(float)
print "Mean P vector negative"
print P_red_mean_n
print "Mean P vector positive"
print P_red_mean_p

# Covar:
P_red_cov_n = np.cov(P_red_n, rowvar=False, ddof=1)
P_red_cov_p = np.cov(P_red_p, rowvar=False, ddof=1)
print "Covar P vector negative"
print P_red_cov_n
print "Covar P vector positive"
print P_red_cov_p

# Histogram:
#calculating number of bins:
#total_min = min(total_p, total_n)
# Assuming half and half
#print P_red.shape[0]
B = int(np.ceil(np.log(P_red.shape[0]/2)/np.log(2))) +1
print "Number of bins:",B

# first number is number of rows and that will be used for first eigenvector
hist_n = np.zeros((B,B),int)
hist_p = np.zeros((B,B),int)

# For translation from previos hw Heigth = 0 and HandSpan = 1
print "Max:",P_red_max
print "Min:",P_red_min

#print P_red_n
for row in P_red_n :
	#print "Negative training set",row
	hist_row = bin_index(row[0],B,P_red_min[0],P_red_max[0])
	hist_col = bin_index(row[1],B,P_red_min[1],P_red_max[1])
	hist_n[hist_row,hist_col]+=1

#print P_red_p
for row in P_red_p :
	#print "Positive training set",row
	hist_row = bin_index(row[0],B,P_red_min[0],P_red_max[0])
	hist_col = bin_index(row[1],B,P_red_min[1],P_red_max[1])
	hist_p[hist_row,hist_col]+=1

print "Negative hist"
print hist_n
print "Positive hist"
print hist_p

'''
# Scatter for homework:
plt.figure(1)
alpha_cloud = 0.1
plt.scatter(P_red[index_n][:,0],P_red[index_n][:,1],color="r",marker="o",alpha=alpha_cloud)
plt.scatter(P_red[index_p][:,0],P_red[index_p][:,1],color="b",marker="o",alpha=alpha_cloud)
plt.axis('equal')
plt.grid()
#plt.show()
plot_histogram(hist_n, hist_p,P_red_max[0],P_red_min[0],P_red_max[1],P_red_min[1])
'''

# Trying 2 samples of each
print np.append(X, T,1)
print index_n
print index_p

sample_idx_n = np.asscalar(np.random.choice(index_n,1))
print "Negative sample:",sample_idx_n

#print X_mean
x_n = X[sample_idx_n]
z_n = x_n - X_mean
p_n = np.dot(z_n,V_t)
p_red_n =p_n[0:2]
#r_n = np.dot(p_n,V_t.T)
#print V_t
#print V_t.T[0:2]
r_n = np.dot(p_red_n,V_t.T[0:2])
xrec_n = (r_n + X_mean)
xrec_pro_n = np.round(np.clip(xrec_n,0,255)).astype(int)
print "Negative feature vec",x_n
print "Cent Neg feature vec",z_n
print "P Neg feature vec",p_n
print "P red Neg feature vec",p_red_n
print "Rec Z Neg feature vec",r_n
print "Rec X Neg feature vec",xrec_n
print "Rec X Neg feature vec",xrec_pro_n

sample_idx_p = np.asscalar(np.random.choice(index_p,1))
print "Positive sample:",sample_idx_p

x_p = X[sample_idx_p]
z_p = x_p - X_mean
p_p = np.dot(z_p,V_t)
p_red_p =p_p[0:2]
r_p = np.dot(p_red_p,V_t.T[0:2])
xrec_p = (r_p + X_mean)
xrec_pro_p = np.round(np.clip(xrec_p,0,255)).astype(int)
#xrec_pro_p = xrec_p
print "Positive feature vec",x_p
print "Cent Pos feature vec",z_p
print "P Pos feature vec",p_p
print "P red Pos feature vec",p_red_p
print "Rec Z Pos feature vec",r_p
print "Rec X Pos feature vec",xrec_p
print "Rec X Pos feature vec",xrec_pro_p

#print P_red


'''
# Plot numbers original and recovered
if (X.shape[1] == 28*28) :
	img_shape = (28,28)
else :
	img_shape = (1, X.shape[1])
plt.subplot(2,2,1)
plt.imshow(x_n.reshape(img_shape),interpolation='None', cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(xrec_pro_n.reshape(img_shape),interpolation='None', cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(x_p.reshape(img_shape),interpolation='None', cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(xrec_pro_p.reshape(img_shape),interpolation='None', cmap=plt.get_cmap('gray'))
plt.show()
'''

#queries

print p_red_n
print hist_n.shape
print hist_n.ndim
print hist_n
print hist_p
print B
hist_row = bin_index(p_red_n[0],B,P_red_min[0],P_red_max[0])
hist_col = bin_index(p_red_n[1],B,P_red_min[1],P_red_max[1])
print hist_row,hist_col
#pos = np.zeros(hist_n.ndim,int)
#print pos
#for i in range(len(pos)) :
#	pos[i] = int(bin_index(p_red_n[i],hist_n.shape[i],P_red_min[i],P_red_max[i]))
print P_red
pos_alt = bin_position(p_red_n, hist_n.shape,P_red_min, P_red_max)
print "Alt",pos_alt
pos = tuple(
	int(bin_index(p_red_n[i],hist_n.shape[i],P_red_min[i],P_red_max[i]))
	for i in range(hist_n.ndim) )
print pos

sam_n = hist_n[pos]
sam_p = hist_p[pos]
print sam_n, sam_p

count_alt = histogram_counts(p_red_n, (hist_n,hist_p),P_red_min, P_red_max)
print "Count alt",count_alt
hist = (hist_n,hist_p)
#hist = hist_n
print type(hist)
counts = [ h[pos] for h in hist ]
#print hist
print counts

res = histogram_clasifier(p_red_n, (hist_n,hist_p),(clabel_n,clabel_p),P_red_min, P_red_max)
print 'sing',res

# Measuring training accuracy:
TP_hist = 0
TN_hist = 0
FP_hist = 0
FN_hist = 0
for i,q in enumerate(P_red) :
	class_output = histogram_clasifier(q, (hist_n,hist_p),(clabel_n,clabel_p),P_red_min, P_red_max)
	ground_truth = np.asscalar(T[i])
	if (ground_truth==clabel_n) :
		if (class_output[0]==clabel_n) :
			status = "True Negative"
			TN_hist +=1
		else :
			status = "False Positive"
			FP_hist +=1
	else :
		if (class_output[0]==clabel_p) :
			status = "True Positive"
			TP_hist +=1
		else :
			status = "False Negative"
			FN_hist +=1
	print class_output, ground_truth, status
del class_output
#print class_output
#s/\(\w\+\)/print '\1',\1
print 'TP_hist',TP_hist
print 'TN_hist',TN_hist
print 'FP_hist',FP_hist
print 'FN_hist',FN_hist

total_hist = TP_hist + TN_hist + FP_hist + FN_hist
accuracy_hist = float(TP_hist + TN_hist)/total_hist

#Verify
if (X.shape[0] != total_hist) :
	print("The accuracy test for histogram is not successful")
	exit(0)

print "Accuracy for histogram",accuracy_hist

#print T



#end of computation


'''
'''
# Writing Excel file:
row = 0

workbook = xlsxwriter.Workbook('results.xlsx')
worksheet = workbook.add_worksheet()

worksheet.set_column('A:A', 30)

worksheet.write(row, 0, 'Roque Alejandro Arcudia Hernandez')
row +=1
worksheet.write(row, 0, 'mu (mean vector)')
for i in range(len(X_mean)) :
	worksheet.write(row, 1+i, X_mean[i])

row +=1
worksheet.write(row, 0, 'v1 (First eigenvector)')
'''
V_t = np.array(
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9],
 [10, 11, 12]]
)
print V_t
print V_t.shape
'''
for i in range(V_t.shape[0]) :
	worksheet.write(row, 1+i, V_t[i,0])

row +=1
worksheet.write(row, 0, 'v2 (Second eigenvector)')
for i in range(V_t.shape[0]) :
	worksheet.write(row, 1+i, V_t[i,1])

row +=2
worksheet.write(row, 0, 'Np (class +1 number of samples)')
worksheet.write(row, 1, total_p)
row +=1
worksheet.write(row, 0, 'Nn (class -1 number of samples)')
worksheet.write(row, 1, total_n)

row +=2
worksheet.write(row, 0, 'mup (class +1 mean vector)')
for i in range(len(P_red_mean_p)) :
	worksheet.write(row, 1+i, P_red_mean_p[i])
row +=1
worksheet.write(row, 0, 'mun (class -1 mean vector)')
for i in range(len(P_red_mean_n)) :
	worksheet.write(row, 1+i, P_red_mean_n[i])

row +=2
worksheet.write(row, 0, 'cp (class +1 covariance matrix)')
'''
P_red_cov_p = np.array(
[[1, 2],
 [3, 4],
 [5, 6]]
)
print P_red_cov_p
print P_red_cov_p.shape
'''
for i in range(P_red_cov_p.shape[1]) :
	for j in range(P_red_cov_p.shape[0]) :
		worksheet.write(row+j, 1+i, P_red_cov_p[j,i])

row +=2
worksheet.write(row, 0, 'cn (class -1 covariance matrix)')
for i in range(P_red_cov_n.shape[1]) :
	for j in range(P_red_cov_n.shape[0]) :
		worksheet.write(row+j, 1+i, P_red_cov_n[j,i])

row +=3
worksheet.write(row, 0, 'Histogram range, pc1 direction')
worksheet.write(row, 1, P_red_min[0])
worksheet.write(row, 2, P_red_max[0])
row +=1
worksheet.write(row, 0, 'Histogram range, pc2 direction')
worksheet.write(row, 1, P_red_min[1])
worksheet.write(row, 2, P_red_max[1])

row +=1
worksheet.write(row, 0, 'Size of histograms')
worksheet.write(row, 1, '25 x 25')

row +=1
worksheet.write(row, 0, 'Hp (class +1 histogram)')
'''
hist_p = np.array(
[[1, 2],
 [3, 4],
 [5, 6]]
)
print hist_p
print hist_p.shape
'''
for i in range(hist_p.shape[1]) :
	for j in range(hist_p.shape[0]) :
		worksheet.write(row+j, 1+i, hist_p[j,i])


row +=26
worksheet.write(row, 0, 'Hn (class -1 histogram)')
for i in range(hist_n.shape[1]) :
	for j in range(hist_n.shape[0]) :
		worksheet.write(row+j, 1+i, hist_n[j,i])

row +=28
worksheet.write(row, 0, 'xp (class +1 feature vector)')
for i in range(len(x_p)) :
	worksheet.write(row, 1+i, x_p[i])
row +=1
worksheet.write(row, 0, 'zp (centered feature vector)')
for i in range(len(z_p)) :
	worksheet.write(row, 1+i, z_p[i])
row +=1
worksheet.write(row, 0, 'pp (2 dim representation)')
for i in range(len(p_red_p)) :
	worksheet.write(row, 1+i, p_red_p[i])
row +=1
worksheet.write(row, 0, 'rp (reconstructed zp)')
for i in range(len(r_p)) :
	worksheet.write(row, 1+i, r_p[i])
row +=1
worksheet.write(row, 0, 'xrecp (reconstructed xp)')
for i in range(len(xrec_pro_p)) :
	worksheet.write(row, 1+i, xrec_pro_p[i])

row +=2
worksheet.write(row, 0, 'xn (class -1 feature vector)')
for i in range(len(x_n)) :
	worksheet.write(row, 1+i, x_n[i])
row +=1
worksheet.write(row, 0, 'zn (centered feature vector)')
for i in range(len(z_n)) :
	worksheet.write(row, 1+i, z_n[i])
row +=1
worksheet.write(row, 0, 'pn (2 dim representation)')
for i in range(len(p_red_n)) :
	worksheet.write(row, 1+i, p_red_n[i])
row +=1
worksheet.write(row, 0, 'rn (reconstructed zn)')
for i in range(len(r_n)) :
	worksheet.write(row, 1+i, r_n[i])
row +=1
worksheet.write(row, 0, 'xrecn (reconstructed xn)')
for i in range(len(xrec_pro_n)) :
	worksheet.write(row, 1+i, xrec_pro_n[i])

row +=4
worksheet.write(row, 0, 'Actual digit represented by xp')
worksheet.write(row, 1, T[sample_idx_p])
row +=1
worksheet.write(row, 0, 'Result of classifying xp using histograms')
row +=1
worksheet.write(row, 0, 'Result of classifying xp using Bayesian')

row +=2
worksheet.write(row, 0, 'Actual digit represented by xn')
worksheet.write(row, 1, T[sample_idx_n])
row +=1
worksheet.write(row, 0, 'Result of classifying xn using histograms')
row +=1
worksheet.write(row, 0, 'Result of classifying xn using Bayesian')

row +=3
worksheet.write(row, 0, 'Training accuracy attained using histograms')
row +=1
worksheet.write(row, 0, 'Training accuracy attained using Bayesian')

'''
worksheet.write(row, 0, 'Hn (class -1 histogram)')
%s/\(.*\)/worksheet.write(row, 0, '\1')
worksheet.write(row, 0, '')

'''


workbook.close()

'''
Questions for prof:
	- In the reconstructed image xrec, should we clip the image if one of the pixels is less than 0 and one is greater than 255. Should we make them integer
	- Hhich probability do we want? the probability of positive or the probability that won the query?

'''

