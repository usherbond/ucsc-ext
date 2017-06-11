import numpy as np

def augment(X) :
	Xa = np.insert(X,0,1,axis=1)
	return Xa

def kernel_aug_vector(xa) :
	#print "Dim:", xa.ndim 
	if xa.ndim != 1 :
		raise ValueError("Array expected to be 1d")
	xa_size = xa.size
	xa_2d = x.reshape((1,xa_size))
	#print "Reshape xa",xa_2d
	prod = np.dot(xa_2d.T,xa_2d)
	#print prod
	idxs = np.triu_indices(xa_size)
	#print idxs
	#print prod[idxs]
	return prod[idxs]

x = np.array([range(1,4)])
print x
print x.shape

mult = np.dot(x.transpose(),x)
print mult

X = np.array(
[[2,3],
 [4,5],
 [6,7]]
)

X = X.reshape(2,3)

print X
print X.shape
#exit()
print X[0:1]
print X[0:1].shape

Xa = augment(X)
print Xa

for x in Xa :
	print x
	xak = kernel_aug_vector(x) 
	print xak

'''
for x in Xa :
	print x
	print "Len:",x.size
	kernel_aug_vector(x) 
	x_2d = x.reshape((1,3))
	print "Reshape",x_2d
	mult = np.dot(x_2d.T,x_2d)
	print mult
	idxs = np.triu_indices(3)
	print idxs
	print mult[idxs]
'''



