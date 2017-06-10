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



np.random.seed(1)

excel_file = 'short.xlsx'
#excel_file = 'Car_Data.xlsx'
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

# Converting to Kesler's form
for i,rec in enumerate(T):
	T_kesler[i, ord_dictionary['Recommendation'][1].index(rec)  ] = 2
	T_index[i, 0] = ord_dictionary['Recommendation'][1].index(rec)
	#T_kesler[i,i%4] = 2
#T_kesler = T_kesler-1
print T_kesler
print T_index

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

classified = linear_classifier(X, W)
print  np.append(T_index,classified,1)

conf_len = len(ord_dictionary['Recommendation'][1])

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
'''
print conf_matrix

conf_total = np.sum(conf_matrix)

PPV = np.zeros(conf_matrix.shape[0],dtype=float)
for i in range(len(PPV)) :
	col_sum = np.sum(conf_matrix[:,i])
	PPV[i] = round(conf_matrix[i,i]/float(col_sum),2)
print "PPV",PPV
max_ppv= max(PPV)

accuracy = np.zeros(conf_matrix.shape[0],dtype=float)
for i in range(len(accuracy)) :
	col_sum = np.sum(conf_matrix[:,i])
	row_sum = np.sum(conf_matrix[i])
	TN = conf_total-col_sum-row_sum+conf_matrix[i,i]
	TP = conf_matrix[i,i]
	#print conf_total, col_sum, row_sum, TN, TP
	accuracy[i] = round((TP+TN)/float(conf_total),2)
print "Accuracy",accuracy

exit()
	

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

P_red = P[:,0:2]
print "First 2 Eigen adjusted data"
print P_red

# This is how we determine which plot to do:
#index_n = np.where(T==clabel_n)[0]
#index_p = np.where(T==clabel_p)[0]
index_n = np.where(T<0)[0]
index_p = np.where(T>0)[0]

P_red_n = P_red[index_n]
P_red_p = P_red[index_p]

total_n = len(index_n)
total_p = len(index_p)
print total_p
print total_n

#uniq = np.unique(T)
uniq = ord_dictionary['Recommendation'][1]

print uniq

Cset = [P_red[np.where(T==cl)[0]] for cl in uniq]
print Cset

print "Negative features:"
print P_red_n
print "Total samples of N :",total_n
print "Positive features:"
print P_red_p
print "Total samples of P :",total_p

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
alpha_cloud = 0.1
for i,cl in enumerate(Cset) :
	plt.scatter(cl[:,0],cl[:,1],color=col[i],marker=mark[i],alpha=alpha_cloud)
#plt.scatter(P_red[index_n][:,0],P_red[index_n][:,1],color="g",marker="o",alpha=alpha_cloud)
#plt.scatter(P_red[index_p][:,0],P_red[index_p][:,1],color="b",marker="o",alpha=alpha_cloud)
plt.axis('equal')
plt.grid()

plt.show()
exit()

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

'''
print P_red_mean_n
print P_red_mean_p
print P_red_cov_n
print P_red_cov_p
print p_red_n
print p_red_p

p_red_n_pdf_n = norm_pdf(p_red_n,P_red_mean_n,P_red_cov_n)
p_red_n_pdf_p = norm_pdf(p_red_n,P_red_mean_p,P_red_cov_p)
print p_red_n_pdf_n
print p_red_n_pdf_p

p_red_n_pcnt_n = total_n * p_red_n_pdf_n
p_red_n_pcnt_p = total_p * p_red_n_pdf_p

print p_red_n_pcnt_n
print p_red_n_pcnt_p

print "Total samples of N (",clabel_n,"):",total_n
print "Total samples of P (",clabel_p,"):",total_p
'''

res_bayes_x_n = bayes_clasifier(p_red_n, (P_red_mean_n,P_red_mean_p), (P_red_cov_n,P_red_cov_p), (total_n,total_p), (clabel_n,clabel_p))
print 'Result for bayes xn',res_bayes_x_n,'supposed to be',T[sample_idx_n]

res_bayes_x_p = bayes_clasifier(p_red_p, (P_red_mean_n,P_red_mean_p), (P_red_cov_n,P_red_cov_p), (total_n,total_p), (clabel_n,clabel_p))
print 'Result for bayes xp',res_bayes_x_p,'supposed to be',T[sample_idx_p]


res_hist_x_n = histogram_clasifier(p_red_n, (hist_n,hist_p),(clabel_n,clabel_p),P_red_min, P_red_max)
print 'Result for histogram xn',res_hist_x_n,'supposed to be',T[sample_idx_n]
res_hist_x_p = histogram_clasifier(p_red_p, (hist_n,hist_p),(clabel_n,clabel_p),P_red_min, P_red_max)
print 'Result for histogram xp',res_hist_x_p,'supposed to be',T[sample_idx_p]

# Measuring training accuracy:
print "Measuring Accuracy for histogram"
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

# Measuring training accuracy bayes:
print "Measuring Accuracy for bayes"
TP_bayes = 0
TN_bayes = 0
FP_bayes = 0
FN_bayes = 0
for i,q in enumerate(P_red) :
	#class_output = histogram_clasifier(q, (hist_n,hist_p),(clabel_n,clabel_p),P_red_min, P_red_max)

	class_output = bayes_clasifier(q, (P_red_mean_n,P_red_mean_p), (P_red_cov_n,P_red_cov_p), (total_n,total_p), (clabel_n,clabel_p))

	ground_truth = np.asscalar(T[i])
	if (ground_truth==clabel_n) :
		if (class_output[0]==clabel_n) :
			status = "True Negative"
			TN_bayes +=1
		else :
			status = "False Positive"
			FP_bayes +=1
	else :
		if (class_output[0]==clabel_p) :
			status = "True Positive"
			TP_bayes +=1
		else :
			status = "False Negative"
			FN_bayes +=1
	print class_output, ground_truth, status
del class_output
#print class_output
#s/\(\w\+\)/print '\1',\1
print 'TP_bayes',TP_bayes
print 'TN_bayes',TN_bayes
print 'FP_bayes',FP_bayes
print 'FN_bayes',FN_bayes

total_bayes = TP_bayes + TN_bayes + FP_bayes + FN_bayes
accuracy_bayes = float(TP_bayes + TN_bayes)/total_bayes

#Verify
if (X.shape[0] != total_bayes) :
	print("The accuracy test for bayes is not successful")
	exit(0)

print "Accuracy for bayes",accuracy_bayes




#print T



#end of computation


'''
'''
# Writing Excel file:
row = 0

workbook = xlsxwriter.Workbook('results.xlsx')
worksheet = workbook.add_worksheet()

worksheet.set_column('A:A', 35)

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
worksheet.write(row, 1, res_hist_x_p[0])
worksheet.write(row, 2, res_hist_x_p[1])
row +=1
worksheet.write(row, 0, 'Result of classifying xp using Bayesian')
worksheet.write(row, 1, res_bayes_x_p[0])
worksheet.write(row, 2, res_bayes_x_p[1])

row +=2
worksheet.write(row, 0, 'Actual digit represented by xn')
worksheet.write(row, 1, T[sample_idx_n])
row +=1
worksheet.write(row, 0, 'Result of classifying xn using histograms')
worksheet.write(row, 1, res_hist_x_n[0])
worksheet.write(row, 2, res_hist_x_n[1])
row +=1
worksheet.write(row, 0, 'Result of classifying xn using Bayesian')
worksheet.write(row, 1, res_bayes_x_n[0])
worksheet.write(row, 2, res_bayes_x_n[1])

row +=3
worksheet.write(row, 0, 'Training accuracy attained using histograms')
worksheet.write(row, 1, accuracy_hist)
row +=1
worksheet.write(row, 0, 'Training accuracy attained using Bayesian')
worksheet.write(row, 1, accuracy_bayes)

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
	-What if we get a singular matrix as covariance

'''

