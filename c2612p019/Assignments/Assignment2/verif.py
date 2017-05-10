import math
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import norm

def bin_index(x, nb, xmin, xmax) :
	"Returns the bin number given the x"
	return	round(float((nb-1)*(x-xmin))/(xmax-xmin))

def bin_center(n, nb, xmin, xmax) :
	"Returns the center of the bin range given the bin number"
	# There is no round to allow this to be numpy array
	return	n*(xmax-xmin)/float(nb-1) + xmin

def bin_bottom(n, nb, xmin, xmax) :
	"Returns the center of the bin range given the bin number"
	# There is no round to allow this to be numpy array
	return	(2*n-1)*(xmax-xmin)/float(2*(nb-1)) + xmin



#def norm_pdf(x, mu, sigma) :
#	return ( math.exp( ((float(x-mu)/sigma)**2)/-2 ) 
#			/ (math.sqrt(2*math.pi)*sigma) )

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

def plot_histogram(m_hist, f_hist,row_max,row_min,col_max,col_min):
	print "Building bar graph"
	row_bins, col_bins = m_hist.shape
	print "row_bins", row_bins
	print "col_bins", col_bins

	row_width = float(row_max-row_min)/(row_bins-1)
	col_width = float(col_max-col_min)/(col_bins-1)
	print "row_width",row_width
	print "col_width",col_width

	m_rows,m_cols = np.nonzero(m_hist)
	m_val = m_hist[m_rows,m_cols]
	f_rows,f_cols = np.nonzero(f_hist)
	f_val = f_hist[f_rows,f_cols]

	fig = plt.figure()
	ax1 = fig.add_subplot(111,projection='3d')
	ax1.bar3d(
		(bin_bottom(f_rows,row_bins,row_min,row_max)),
		(bin_bottom(f_cols,col_bins,col_min,col_max)),
		np.zeros(len(f_val),int),
		row_width,
		col_width,
		f_val,
		color='r',
		alpha=0.5)

	ax1.bar3d(
		(bin_bottom(m_rows,row_bins,row_min,row_max)),
		(bin_bottom(m_cols,col_bins,col_min,col_max)),
		np.zeros(len(m_val),int),
		row_width,
		col_width,
		m_val,
		color='b',
		alpha=0.5)

	plt.show()

# Creating random set:
p_female_rnd = 0.532934131737
#rand_set_size = 5000000
rand_set_size = 2000000
rnd_num_bins = int(np.ceil(np.log(rand_set_size/2)/np.log(2))) +1
print rnd_num_bins 
uniform_dist = np.random.uniform(size=rand_set_size)
#print uniform_dist
rnd_gender = ["Male" if x > p_female_rnd else "Female" for x in uniform_dist]
#print rnd_gender
#print rnd_gender.count("Female")
print "Verify female prob:",rnd_gender.count("Female")/float(rand_set_size)

male_means_rnd = np.array( [ 71.28846154,  22.30128205] )
female_means_rnd = np.array( [ 65.25280899,  19.6011236 ] )
#female_means_rnd = np.array( [ 5.25280899,  2.6011236 ] )
male_covar_rnd = np.matrix( [[ 7.08778721,  1.80157343],
		 [ 1.80157343,  2.06064769]] )
female_covar_rnd = np.matrix([[ 7.75780452,  1.65170135],
		 [ 1.65170135,  1.75670327]])

feature_matrix = [np.random.multivariate_normal(male_means_rnd,male_covar_rnd) if x =="Male" 
				else np.random.multivariate_normal(female_means_rnd,female_covar_rnd) for x in rnd_gender]

#print feature_matrix
rnd_height = [row[0] for row in feature_matrix]
rnd_handspan = [row[1] for row in feature_matrix]
print rnd_height
print rnd_handspan
#Sex  Height  HandSpan
d = {'Sex':rnd_gender,'Height':rnd_height,'HandSpan':rnd_handspan}
random_df = pd.DataFrame(d)
#print random_df
#exit()
'''
#0.532934
height_array = [np.random.normal(male_mean,male_std) if x =="Male" 
				else np.random.normal(female_mean,female_std) for x in gender_array]
print height_array

d = {'Gender':gender_array,'Total_Inches':height_array}
random_df = pd.DataFrame(d)
print random_df
exit()
'''

head_num = 50;

#excel_file = 'short.xlsx'
excel_file = 'Assignment_2_Data_and_Template.xlsx'


#data_frame = pd.read_csv('Height.csv')
#data_frame = pd.read_excel('Assignment_2_Data_and_Template.xlsx',sheetname='Data')
#data_frame = pd.read_excel(excel_file,sheetname='Data')
#data_frame = pd.read_excel('dummy.xlsx',sheetname='Data')
#print data_frame.query('Sex=="Female"')
data_frame = random_df
print data_frame
#exit()
#queries = pd.read_excel('short.xlsx',sheetname='Queries',header=1,names=['Height','Handspan'] )
queries = pd.read_excel(excel_file,sheetname='Queries',header=1)[['Height','Handspan']]
#print queries
#exit()

min_height = data_frame["Height"].min()
max_height = data_frame["Height"].max()
min_handspan = data_frame["HandSpan"].min()
max_handspan = data_frame["HandSpan"].max()

print "Max height:",max_height
print "Min height:",min_height
print "Max handspan",max_handspan
print "Min handspan",min_handspan

#print max_height-min_height 
#print max_handspan-min_handspan

# with the assignment sprteadsheet the range height bins 22 and handspan bins 20 creates an array where all the integer values are in the middle or the 0.5 values in case of the handspan

#height_bins = int(max_height-min_height + 1)
#height_bins = 22
height_bins = 8
#handspan_bins = int(max_handspan-min_handspan +1)
#handspan_bins = 20
handspan_bins = 8

height_bins = rnd_num_bins 
handspan_bins = rnd_num_bins 

print "Height bins:",height_bins
print "Handspan bins:",handspan_bins

height_width = float(max_height-min_height)/(height_bins-1)
handspan_width = float(max_handspan-min_handspan)/(handspan_bins-1)

print "height_width", height_width
print "Handspan width", handspan_width

#height_bins = 17
#handspan_bins = 14
#height_bins = 9
#handspan_bins = 4
#handspan_bins = 3

male_hist = np.zeros((height_bins,handspan_bins),int)
female_hist = np.zeros((height_bins,handspan_bins),int)

# Hist:
male_df = data_frame.query('Sex=="Male"')
female_df = data_frame.query('Sex=="Female"')

#print male_df
#print female_df
male_n = male_df['Sex'].count()
female_n = female_df['Sex'].count()
total_n = female_n+male_n
p_female = float(female_n)/total_n
print "Number of males:",male_n
print "Number of females:",female_n
print "Total",total_n
print "Probability of Female:",p_female

#print male_df[['Height','HandSpan']]
features = ['Height','HandSpan']
#print male_df[features].as_matrix().mean(axis=0)
#print features
#male_means = male_df.mean()
#female_means = female_df.mean()
male_means = male_df[features].as_matrix().mean(axis=0)
female_means = female_df[features].as_matrix().mean(axis=0)

#numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)[source]
male_covar = np.cov(male_df[features], rowvar=False, ddof=1)
female_covar = np.cov(female_df[features], rowvar=False, ddof=1)

print "Male means",male_means
print "Female means",female_means

print "Male covar",male_covar
print "Female covar",female_covar

#male_covar_inv = np.linalg.inv(male_covar)
#female_covar_inv = np.linalg.inv(female_covar)

#print male_covar_inv
#print female_covar_inv
#x = [72,21]
#x = np.array([0,0])
#mu = np.array([1,1])
#sigma = [[9,1],[1,4]]
#p = norm_pdf(x,male_means,male_covar)
#p = norm_pdf(x,mu,sigma)
#print p

#mul = np.matmul(male_covar,male_covar_inv)
#mul = np.matrix(male_covar)*np.matrix(male_covar_inv)
#mul = np.matrix(female_covar)*np.matrix(female_covar_inv)
#print mul



#exit()

for index, row in male_df.iterrows():
	#print index, row
	hist_row = bin_index(row["Height"],height_bins,min_height,max_height)
	hist_col = bin_index(row["HandSpan"],handspan_bins,min_handspan,max_handspan)
	#male_hist[hist_row,hist_col] = hist_row*handspan_bins + hist_col + 1
	male_hist[hist_row,hist_col]+=1

print male_hist

for index, row in female_df.iterrows():
	#print index, row
	hist_row = bin_index(row["Height"],height_bins,min_height,max_height)
	hist_col = bin_index(row["HandSpan"],handspan_bins,min_handspan,max_handspan)
	#female_hist[hist_row,hist_col] = hist_row*handspan_bins + hist_col + 2
	female_hist[hist_row,hist_col]+=1

print female_hist

y = np.asfarray(range(height_bins))
# This is the bottom of the range but thats how the graph comes
#y_ticks = np.around(bin_center(y,height_bins,min_height,max_height)-height_width/2,1)
y_ticks = np.around(bin_bottom(y,height_bins,min_height,max_height),2)
print y_ticks
print np.around(bin_center(y,height_bins,min_height,max_height),2)

x = np.asfarray(range(handspan_bins))
x_ticks = np.around(bin_center(x,handspan_bins,min_handspan,max_handspan),2)
print x_ticks
print np.around(bin_bottom(x,handspan_bins,min_handspan,max_handspan),2)

# Reconstructing histogram from normal:
#male_hist_rec = np.zeros((height_bins,handspan_bins),int)
#female_hist_rec = np.zeros((height_bins,handspan_bins),int)
male_hist_rec = np.zeros((height_bins,handspan_bins))
female_hist_rec = np.zeros((height_bins,handspan_bins))
for row in range(height_bins) :
	for col in range(handspan_bins) :
		heigth = bin_center(row,height_bins,min_height,max_height)
		handspan = bin_center(col,handspan_bins,min_handspan,max_handspan)
		male_pdf = norm_pdf([heigth,handspan],male_means,male_covar)
		female_pdf = norm_pdf([heigth,handspan],female_means,female_covar)
		male_hist_rec[row,col]=male_n*male_pdf*height_width*handspan_width
		female_hist_rec[row,col]=female_n*female_pdf*height_width*handspan_width

print np.round(male_hist_rec)
print np.round(female_hist_rec)

#exit()

# Fix in the formatting of Handspan that has leading spaces
queries['Handspan'] = queries['Handspan'].str.strip()
for index, row in queries.iterrows():
	#print row.as_matrix()
	rownum = pd.to_numeric(row).astype(float)
	print rownum
	#xval = pd.to_numeric(row).astype(float).as_matrix()
	print rownum.as_matrix()
	hist_row = bin_index(rownum["Height"],height_bins,min_height,max_height)
	hist_col = bin_index(rownum["Handspan"],handspan_bins,min_handspan,max_handspan)
	print "row",hist_row
	print "col",hist_col
	male_sample = male_hist[hist_row,hist_col]
	female_sample = female_hist[hist_row,hist_col]
	print "male_sample",male_sample
	print "female_sample",female_sample

	try :
		p_female_hist = ( float(female_sample)
				/ (female_sample+male_sample) )
	except ZeroDivisionError as err:
		print "WARNING:",err
		p_female_hist = float('NaN')
	print "prob hist",p_female_hist

	male_pdf = norm_pdf(rownum.as_matrix(),male_means,male_covar)
	female_pdf = norm_pdf(rownum.as_matrix(),female_means,female_covar)
	p_female_norm = ( float(female_n*female_pdf)
			/ (female_n*female_pdf+male_n*male_pdf) )
	print "prob norm",p_female_norm
	

#exit()

plot_histogram(male_hist, female_hist,max_height,min_height,max_handspan,min_handspan)
plot_histogram(np.round(male_hist_rec,1), np.round(female_hist_rec,1),max_height,min_height,max_handspan,min_handspan)


'''

#z = np.zeros((height_bins,handspan_bins),int)
#for i in x :
#	for j in y :
#		z[j,i] = j*handspan_bins + i + 1
#print z

#male_hist[2,1] = 2
male_rows,male_cols = np.nonzero(male_hist)
male_val = male_hist[male_rows,male_cols]
print male_rows,male_cols
print male_val

#female_hist[2,1] = 5
female_rows,female_cols = np.nonzero(female_hist)
female_val = female_hist[female_rows,female_cols]
print female_rows,female_cols
print female_val

#row, col = male_hist.shape
#print row
#print col


#y = np.asfarray(range(height_bins))
#x = np.asfarray(range(handspan_bins))

#y_ticks = np.around(bin_center(y,height_bins,min_height,max_height)-height_width/2,1)
#x_ticks = np.around(bin_center(x,handspan_bins,min_handspan,max_handspan)-handspan_width/2,1)


fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
#y_ticks = np.around(bin_bottom(y,height_bins,min_height,max_height),1)
ax1.bar3d(
	(bin_bottom(female_rows,height_bins,min_height,max_height)),
	(bin_bottom(female_cols,handspan_bins,min_handspan,max_handspan)),
	np.zeros(len(female_val),int),
	height_width,
	handspan_width,
	female_val,
	color='r',
	alpha=0.5)

ax1.bar3d(
	(bin_bottom(male_rows,height_bins,min_height,max_height)),
	(bin_bottom(male_cols,handspan_bins,min_handspan,max_handspan)),
	np.zeros(len(male_val),int),
	height_width,
	handspan_width,
	male_val,
	color='b',
	alpha=0.5)

plt.show()

#print male_df
#print female_df
#print data_frame.query('Sex=="Male"') 
#for data in data_frame.query('Sex=="Male"') :
#	print data
	#male_bin_array[bin_index(data,num_bins,min_height,max_height)]+=1

'''


'''
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
X, Y = np.meshgrid(x, y)
Zhandspan_binshandspan_bino = np.zeros((height_bins,handspan_bins),int)
print X.ravel()
print Y
print Z

#ax1.plot_surface(X, Y, Z)
#dx = np.ones((height_bins,handspan_bins))/2
#dy = np.ones((height_bins,handspan_bins))/2
dx = 1
dy = 1
dz = z
ax1.bar3d(X.ravel(), Y.ravel(), Z.ravel(), dx, dy, dz.ravel(), color='b',alpha=0.5)

secx= [1,2]
secy =[2,3]
secz =[0,0]
valz =[20,21]
valz2 =[3,5]
ax1.bar3d(secx, secy, secz, 1, 1, valz, color='r',alpha=0.5)
plt.show()
'''




