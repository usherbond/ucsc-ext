import pandas as pd
#import matplotlib as plt
import matplotlib.pyplot as plt
#from numpy import append, array, int8, uint8, zeros as np
import numpy as np

def linear_classifier(x_vectors, W) :
	aug_ones = np.ones((x_vectors.shape[0],1),dtype=float)
	#print aug_ones
	#print aug_ones.shape
	#print type(aug_ones)
	xa_vectors = np.append(aug_ones,x_vectors,1)
	#print xa_vectors

	classify_failure = np.dot(xa_vectors, W)
	#classify_failure[2,0] = 0
	#print classify_failure
	results = np.zeros((x_vectors.shape[0],1),dtype=int)
	#print results
	#print classify_failure.shape[1]
	if (classify_failure.shape[1]==1) :
		for i,cl in enumerate(classify_failure) :
			#print i, np.asscalar(cl)
			cl_scal = np.asscalar(cl)
			if cl_scal == 0 :
				print "ERROR: 0 value in output class"
				exit(1)
			results[i,0] = 1 if cl_scal>0 else -1
	else :
		for i,cl in enumerate(classify_failure) :
			max_val = max(cl)
			#print i, cl, max_val
			idxs = np.where(cl == max_val)[0]
			if (len(idxs) == 1) :
				#print idxs
				#print np.asscalar(idxs)
				results[i,0] = np.asscalar(idxs)
			else :
				print "ERROR : several max indexes:",idxs
				exit(1)

	return results


#seed 5 looks promissing
np.random.seed(5)

#excel_file = 'short.xlsx'
excel_file = 'Car_Data.xlsx'
#excel_file = 'Assignment_4_Data_and_Template.xlsx'

data_frame = pd.read_excel(excel_file,sheetname='Sheet1',skiprows=1)
print data_frame
#print data_frame['Price'].unique()

# this one seems to be in order:
col_names = list(data_frame)
cur_offset = 0
ord_dictionary = {}
for name in col_names :
	ord_list = data_frame[name].unique()
	print name,":",ord_list,len(ord_list)
	ord_dictionary[name] = (cur_offset,list(ord_list))
	cur_offset += len(ord_list)

#print ord_dictionary
#print ord_dictionary['Recommendation']
set_rows = data_frame['Recommendation'].count()
print ord_dictionary['Recommendation'][0]
# Recomendation is T, its offset is the number of columns we need:
X = np.zeros((set_rows,ord_dictionary['Recommendation'][0]),int)
#print X

for index, row in data_frame.iterrows():
	for col in col_names :
		if (col != 'Recommendation') :
			item = row[col]
			idx = ord_dictionary[col][1].index(item)
			off = ord_dictionary[col][0]
			#print index, item, idx, off
			X[index,idx + off] = 1

T = np.array(data_frame['Recommendation'])

print X
print T
#print T.shape

T_kesler = np.zeros((T.shape[0],len(ord_dictionary['Recommendation'][1])),dtype=int)
T_index = np.zeros((T.shape[0],1),dtype=int)
T_binary = np.zeros((T.shape[0],1),dtype=int)

# Converting to Kesler's form
for i,rec in enumerate(T):
	T_kesler[i, ord_dictionary['Recommendation'][1].index(rec)  ] = 2
	T_index[i, 0] = ord_dictionary['Recommendation'][1].index(rec)
	T_binary[i,0] = -1 if rec=='unacc' else 1
	#T_kesler[i,i%4] = 2
T_kesler = T_kesler-1
print T_kesler
print T_index
print T_binary

# Generating Xa and its pesudo inverse
#X0 = np.ones((X.shape[0],1),dtype=float)
X0 = np.ones((X.shape[0],1),dtype=int)
#print X0
#print X0.shape
#print type(X0)
Xa = np.append(X0,X,1)
print Xa
print Xa.shape
Xapinv = np.linalg.pinv(Xa)
print Xapinv
print Xapinv.shape


W = np.dot(Xapinv, T_kesler)
W_binary = np.dot(Xapinv, T_binary)


'''
W_type = np.array(
[[9,10,11,12],
 [5, 6, 7, 8],
 [1, 2, 3, 4]]
)
'''
print "W"
print W
print W.shape
print "W binary"
print W_binary
print W_binary.shape

classified = linear_classifier(X, W)
print  np.append(T_index,classified,1)

classified_bin = linear_classifier(X, W_binary)
print  np.append(T_binary,classified_bin,1)

conf_len = len(ord_dictionary['Recommendation'][1])

'''
conf_matrix = np.array(
[[25,28,73,38,68, 0],
 [ 0, 0,57,74, 0,68],
 [28, 0, 0,66,97,61],
 [46, 0,29,64,53,41],
 [70,88,39, 0,30,86],
 [ 0,40,83, 0,64,58]]
)
'''
conf_matrix = np.zeros((conf_len,conf_len),dtype=int)
for i, true_cl in enumerate(T_index) :
	tcl = np.asscalar(true_cl)
	cas = np.asscalar(classified[i])
	#print i, tcl, cas
	conf_matrix[tcl,cas] += 1
	#row_idx = (tcl + 1)/2
print "Confusion matrix"
print conf_matrix
'''
golden:
[[1191   19    0    0]
 [ 126  258    0    0]
 [   0   65    0    0]
 [   0   69    0    0]]
'''
# Confuson binary
conf_matrix_bin = np.zeros((2,2),dtype=int)
for i, true_cl in enumerate(T_binary) :
	tcl = (1-np.asscalar(true_cl))/2
	cas = (1-np.asscalar(classified_bin[i]))/2
	#print i, tcl, cas
	conf_matrix_bin[tcl,cas] += 1
	#row_idx = (tcl + 1)/2
print "Confusion matrix binary"
print conf_matrix_bin



conf_total = np.sum(conf_matrix)

PPV = np.zeros(conf_matrix.shape[0],dtype=float)
for i in range(len(PPV)) :
	col_sum = np.sum(conf_matrix[:,i])
	PPV[i] = round(conf_matrix[i,i]/float(col_sum),2)
print "PPV",PPV
max_ppv= max(PPV)
#1728
PPV_bin = conf_matrix_bin[0,0]/float(np.sum(conf_matrix_bin[:,0]))
print "PPV binary",PPV_bin

accuracy = np.zeros(conf_matrix.shape[0],dtype=float)
for i in range(len(accuracy)) :
	col_sum = np.sum(conf_matrix[:,i])
	row_sum = np.sum(conf_matrix[i])
	TN = conf_total-col_sum-row_sum+conf_matrix[i,i]
	TP = conf_matrix[i,i]
	#print conf_total, col_sum, row_sum, TN, TP
	accuracy[i] = round((TP+TN)/float(conf_total),2)
print "Accuracy",accuracy
accuracy_bin = np.trace(conf_matrix_bin)/float(np.sum(conf_matrix_bin))
print "Accuracy binary",accuracy_bin
exit()

# K means classification:

# initializations:
K = 2

X_max = np.amax(X,axis=0)
X_min = np.amin(X,axis=0)
X_diff = (X_max-X_min).astype(float)

print X
print X_max,X_min
print X_diff
print len(X_diff)

# Be careful with the type not being a matrix
mu = (np.random.rand(K,len(X_diff)) * X_diff) + X_min
print np.round(mu,1)

C = np.zeros((X.shape[0],1),int)
#print C.shape
#print  np.append(T_index,C,1)
error = 10000.0 ; tolerance = 0 ; iterations = 0

while ((error>tolerance) and (iterations < 10000)) :
	for i in range(X.shape[0]) :
		#print X[i]
		mu_minus_x = mu - X[i]
		#print np.round(mu_minus_x,1)
		# this is fast but wasteful:
		mult = np.dot(mu_minus_x,mu_minus_x.transpose())
		#print mult
		diag = np.diagonal(mult)
		#print diag
		C[i,0] = np.argmin(diag)
		#print C[i,0]

	newmu = np.zeros((K,len(X_diff)),float)
	for k in range(K) :
		k_index = np.where(C==k)[0]
		#print k_index
		#print X[k_index]
		if  (len(k_index)>0) :
			newmu[k] = np.mean(X[k_index],axis=0).astype(float) #Avoiding potential issues with int
			#print newmu[k]
		else :
			# controversial but use the old one:
			newmu[k] = mu[k]

	#print newmu
	#print mu
	mu_err = mu - newmu
	#print mu_err
	mu_err_abs = np.abs(mu_err)
	#print mu_err_abs
	error =  np.amax(mu_err_abs)
	print error

	mu = newmu
	iterations += 1
	#print mu

print  np.append(T_index,C,1)
print  iterations, error
print C
print np.bincount(C.flatten())

for k in range(K) :
	k_index = np.where(C==k)[0]
	print "Composition of mean k =",k
	#print T_index[k_index]
	bins = np.bincount(T_index[k_index].flatten())
	print bins

#X_mean = np.mean(X,axis=0).astype(float) #Avoiding potential issues with int

#index_n = np.where(T==clabel_n)[0]

print mu

	

'''
Golden:
Price : [u'vhigh' u'high' u'med' u'low']
Maintenance : [u'vhigh' u'high' u'med' u'low']
Doors : [2 3 4 u'5more']
Persons : [2 4 u'more']
Trunk : [u'small' u'med' u'big']
Safety : [u'low' u'med' u'high']
Recommendation : [u'unacc' u'acc' u'vgood' u'good']
'''
#X = df_matrix[:,:-2]
#print X
#T = df_matrix[:,-1:]
#print T

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
#
print "First vector"
print V_t[:,0]
print np.round(V_t[:,0],2)

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

# Adding the new k mean to the plot
Pmu = np.dot(mu - X_mean, V_t)
#print "Transformed", Pmu

P_red = P[:,0:2]
print "First 2 Eigen adjusted data"
print P_red

Pmu_red = Pmu[:,0:2]

# This is how we determine which plot to do:


#uniq = np.unique(T)
uniq = ord_dictionary['Recommendation'][1]

print uniq

Cset = [P_red[np.where(T==cl)[0]] for cl in uniq]
print Cset

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

# Scatter for homework:
# other colors not rgb: y, c, m
col = ['r','b','g','y','c','m']
mark = ['_','|','x','o','|','x']
#col = ['r','b']
alpha_cloud = 0.9
for i,cl in enumerate(Cset) :
	plt.scatter(cl[:,0],cl[:,1],color=col[i],marker=mark[i],alpha=alpha_cloud)
plt.scatter(Pmu_red[:,0],Pmu_red[:,1],color="k",marker="s",alpha=1)
plt.axis('equal')
plt.grid()

plt.show()
exit()

