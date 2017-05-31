import math
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import norm

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



#excel_file = 'short.xlsx'
#excel_file = 'Assignment_2_Data_and_Template.xlsx'
excel_file = 'short_a4.xlsx'

#excel_file = 'Assignment_4_Data_and_Template.xlsx'

data_frame = pd.read_excel(excel_file,sheetname='Training Data')
print data_frame
df_matrix = data_frame.as_matrix()
# max eig:
#df_matrix_sub = data_frame.as_matrix(["y-acc","Load","Failure","Type"])
#df_matrix_sub = data_frame.as_matrix(["Temperature","x-acc","Failure","Type"])
df_sub = data_frame
df_sub["acc"] = df_sub["x-acc"]+df_sub["y-acc"]+df_sub["z-acc"]
# This one gives 3 colors:
#df_matrix_sub = df_sub.as_matrix(["Temperature","z-acc","Failure","Type"])
# this one gives even better results:
df_matrix_sub = df_sub.as_matrix(["Temperature","acc","Failure","Type"])
print df_matrix
print df_matrix_sub
df_matrix = df_matrix_sub

X = df_matrix[:,:-2]
print X
T_type = df_matrix[:,-1:]
T_failure = df_matrix[:,-2:-1]

print "Data set"
print X
print "Labels"
print T_failure
print T_type

T_type_kesler = np.zeros((T_type.shape[0],6),dtype=float)

# Converting to Kesler's form
for i,typ in enumerate(T_type):
	T_type_kesler[i,np.asscalar(typ)] = 2.0
T_type_kesler = T_type_kesler-1
print T_type_kesler

# Generating Xa and its pesudo inverse
X0 = np.ones((X.shape[0],1),dtype=float)
print X0
print X0.shape
print type(X0)
Xa = np.append(X0,X,1)
print Xa
Xapinv = np.linalg.pinv(Xa)
print Xapinv

# Calculate the w's
W_failure = np.dot(Xapinv, T_failure)
print "W failure"
print W_failure

W_type = np.dot(Xapinv, T_type_kesler)
'''
W_type = np.array(
[[9,10,11,12],
 [5, 6, 7, 8],
 [1, 2, 3, 4]]
)
'''
print "W type"
print W_type

queries_df = pd.read_excel(excel_file,sheetname='To be classified',skiprows=3)
print queries_df
queries_df_matrix = queries_df.as_matrix()
print queries_df_matrix
queries_df_sub = queries_df
queries_df_sub["acc"] = queries_df_sub["x-acc"]+queries_df_sub["y-acc"]+queries_df_sub["z-acc"]
# this one gives even better results:
queries_df_matrix_sub = queries_df_sub.as_matrix(["Temperature","acc","Failure","Type"])
print queries_df_matrix
print queries_df_matrix_sub
queries_df_matrix = queries_df_matrix_sub
X_queries = queries_df_matrix[:,:-2]
print X_queries

#results_failure = linear_classifier(X_queries, W_failure)
#X_queries = X
results_failure = linear_classifier(X_queries, W_failure)
print results_failure
results_type = linear_classifier(X_queries, W_type)
print results_type

# Test accuracy:
cl_as_failure = linear_classifier(X, W_failure)
cl_as_type = linear_classifier(X, W_type)


# From this point it is plotting related
line_failure = -1*W_failure[:-1]/W_failure[-1,0]
print "Line failure"
print line_failure

line_type = -1*W_type[:-1]/W_type[-1].astype(float)
print "Line type"
print line_type


Xa_max = np.amax(Xa,axis=0)
Xa_min = np.amin(Xa,axis=0)
print Xa_max
print Xa_min
Xrange = np.array([Xa_min,Xa_max])
#exit()
print Xrange
print Xrange.shape
Xpoints = Xrange[:,:-1]
print Xpoints
Xd_failure = np.dot(Xpoints,line_failure)
Xd_type = np.dot(Xpoints,line_type)
print Xrange[:,1]
print Xd_failure
print Xd_type
#


# ploting:
uniq_failure = np.unique(T_failure)
print uniq_failure
Cset_failure = [X[np.where(T_failure==cl)[0]] for cl in uniq_failure]
print Cset_failure

uniq_type = np.unique(T_type)
print uniq_type
Cset_type = [X[np.where(T_type==cl)[0]] for cl in uniq_type]
print Cset_type

col = ['r','b','g','y','c','m']
alpha_cloud = 0.1

plt.subplot(1,2,1)
for i,cl in enumerate(Cset_failure) :
	plt.scatter(cl[:,0],cl[:,1],color=col[i],marker="o",alpha=alpha_cloud)
#plt.plot([60,61],[22,20],"k")
plt.plot(Xrange[:,1],Xd_failure,"k")
for i, x_q in enumerate(X_queries) :
	#print i, x_q
	plt.plot(x_q[0],x_q[1],"rs" if results_failure[i] < 0 else "bs")
#results_failure = linear_classifier(X_queries, W_failure)
#print results_failure
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
#plt.show()

plt.subplot(1,2,2)
for i,cl in enumerate(Cset_type) :
	plt.scatter(cl[:,0],cl[:,1],color=col[i],marker="o",alpha=alpha_cloud)
#plt.plot([60,61],[22,20],"k")
#plt.plot(Xrange[:,1],Xd,"k")
for i in uniq_type :
	plt.plot(Xrange[:,1],Xd_type[:,i],col[i])
for i, x_q in enumerate(X_queries) :
	#print i, x_q
	#print results_type[i,0]
	plt.plot(x_q[0],x_q[1],col[results_type[i,0]]+'s')
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
plt.show()
exit()


'''
# calculating parameters for plotting :
line_failure = -1*W_failure[:-1]/W_failure[-1,0]
#line = np.array([16,1])
Xa_max = np.amax(Xa,axis=0)
Xa_min = np.amin(Xa,axis=0)
print Xa_max
print Xa_min
Xrange = np.array([Xa_min,Xa_max])
print line_failure
#exit()
print Xrange
print Xrange.shape
Xpoints = Xrange[:,:-1]
print Xpoints
Xd = np.dot(Xpoints,line_failure)
print Xrange[:,1]
print Xd
#

# ploting:
uniq = np.unique(T_failure)
print uniq
Cset = [X[np.where(T_failure==cl)[0]] for cl in uniq]
print Cset


col = ['r','b','g','y','c','m']
alpha_cloud = 0.1
#plt.subplot(2,2,1)
for i,cl in enumerate(Cset) :
	plt.scatter(cl[:,0],cl[:,1],color=col[i],marker="o",alpha=alpha_cloud)
#plt.plot([60,61],[22,20],"k")
plt.plot(Xrange[:,1],Xd,"k")
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
plt.show()
'''


exit()
#queries = pd.read_excel('short.xlsx',sheetname='Queries',header=1,names=['Height','Handspan'] )
queries = pd.read_excel(excel_file,sheetname='Queries',header=1)[['Height','Handspan']]
print queries


features = ['Height','HandSpan']

print data_frame[features]
X = np.array(data_frame[features])
#T = np.array([ 1.0 if cl == 'Male' else -1.0 for cl in data_frame['Sex']])
#T = np.array(data_frame['Sex'])
T = np.zeros((data_frame['Sex'].count(),1),dtype=float)
for i,cl in enumerate(data_frame['Sex']) :
	T[i] = 1.0 if cl == 'Male' else -1.0
print X
print X.shape
print T
print T.shape
print type(T)
X0 = np.ones((X.shape[0],1),dtype=float)
print X0
print X0.shape
print type(X0)
Xa = np.append(X0,X,1)
print Xa
Xapinv = np.linalg.pinv(Xa)
print Xapinv

# Calculate the w's
W = np.dot(Xapinv, T)
print W
line = -1*W[:-1]/W[-1,0]
#line = np.array([16,1])
Xa_max = np.amax(Xa,axis=0)
Xa_min = np.amin(Xa,axis=0)
print Xa_max
print Xa_min
Xrange = np.array([Xa_min,Xa_max])
print line
#exit()
print Xrange
print Xrange.shape
Xpoints = Xrange[:,:-1]
print Xpoints
Xd = np.dot(Xpoints,line)
print Xrange[:,1]
print Xd
#exit()

index_n = np.where(T<0)[0]
index_p = np.where(T>0)[0]
print index_n
print index_p


alpha_cloud = 0.5
#plt.subplot(2,2,1)
plt.scatter(X[index_n][:,0],X[index_n][:,1],color='r',marker="o",alpha=alpha_cloud)
plt.scatter(X[index_p][:,0],X[index_p][:,1],color='b',marker="o",alpha=alpha_cloud)
#plt.plot([60,61],[22,20],"k")
plt.plot(Xrange[:,1],Xd,"k")
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
plt.show()




