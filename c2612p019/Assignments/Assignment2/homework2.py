import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def bin_index(x, nb, xmin, xmax) :
	"Returns the bin number given the x"
	return	round(float((nb-1)*(x-xmin))/(xmax-xmin))

def bin_center(n, nb, xmin, xmax) :
	"Returns the center of the bin range given the bin number"
	# There is no round to allow this to be numpy array
	return	n*(xmax-xmin)/(nb-1) + xmin

def norm_pdf(x, mu, sigma) :
	return ( math.exp( ((float(x-mu)/sigma)**2)/-2 ) 
			/ (math.sqrt(2*math.pi)*sigma) )

### From this point it should be a proceadure:
#inputs:
# - data frame
# - min_height
# - max_height
# - num_bins
def compute_probability_proc(data_frame, min_height, max_height, num_bins) :
	male_df = data_frame[data_frame["Gender"].str.contains("Male")]
	female_df = data_frame[data_frame["Gender"].str.contains("Female")]

	#print male_df
	#print female_df

	male_bin_array = np.zeros(num_bins,int)
	female_bin_array = np.zeros(num_bins,int)

	# Stats for male:
	for data in male_df['Total_Inches']:
		male_bin_array[bin_index(data,num_bins,min_height,max_height)]+=1
	male_mean = np.mean(male_df['Total_Inches'])
	male_std = np.std(male_df['Total_Inches'],ddof=1)
	#male_n = male_df['Total_Inches'].size
	male_n = male_df['Total_Inches'].size

	#print male_bin_array

	# Stats for female:
	for data in female_df['Total_Inches']:
		female_bin_array[bin_index(data,num_bins,min_height,max_height)]+=1
	female_mean = np.mean(female_df['Total_Inches'])
	female_std = np.std(female_df['Total_Inches'],ddof=1)
	female_n = female_df['Total_Inches'].size

	#print female_bin_array

	print ','.join(map(str, female_bin_array)) 
	print ','.join(map(str, male_bin_array)) 

	print "Female mean:,",female_mean,", Female std:,",female_std,", Female N:,",female_n
	print "Male mean:,",male_mean,", Male std:,",male_std,", Male N:,",male_n

	queries = np.array([55, 60, 65, 70, 75, 80])
	#queries = np.array([55, 65, 70, 75, 80])
	#queries = np.array([58])
	#print queries
	hist_str = "Hist"
	bayess_str = "Bayess"
	for i in queries:
		idx = bin_index(i,num_bins,min_height,max_height)
		#print idx
		if (idx < 0) :
			p_female_hist = 1.0
		elif (idx < 0 or idx > (num_bins-1)):
			p_female_hist = 0.0
		else:
			try :
				total_n = ( male_bin_array[idx] 
						+ female_bin_array[idx] )
				p_female_hist = ( float(female_bin_array[idx]) 
						/ total_n)
			except ZeroDivisionError as err:
				print "WARNING:",err
				p_female_hist = float('NaN')
		#print "Probability of female given a height of",i, \
		#      "in bin",idx+1,"is",p_female_hist

		female_pdf = norm.pdf(i,female_mean,female_std)
		male_pdf = norm.pdf(i,male_mean,male_std)
		p_female_norm = ( float(female_n*female_pdf)
				/ (female_n*female_pdf+male_n*male_pdf) )
		#print "P of female given a height of",i, \
		#      "with mean",round(female_mean,1),"and std of", \
		#      round(female_std,1),"is",round(p_female_norm,3)

		#print "P of female given a height of",i, \
		#      "Histogram",round(p_female_hist,4), \
		#      "Bayess",round(p_female_norm,4)
		#print "P of female given a height of {:2d} Histogram {:.4f} Bayess {:.4f}".format(i,p_female_hist,p_female_norm)
		error = abs(p_female_hist - p_female_norm)
		print ("P of female given a height of, {:2d}," + 
			" Histogram, {:.4f}, Bayess, {:.4f}, Error, {:.4f}").format(
				i,p_female_hist,p_female_norm,error)
		hist_str += ",{:.4f}".format(p_female_hist)
		bayess_str += ",{:.4f}".format(p_female_norm)

	print hist_str
	print bayess_str


	#Gausian Graph:
	bar_width = float(max_height-min_height)/(num_bins-1)
	samples = num_bins
	width = float(max_height-min_height)/(samples-1)
	x_norm = np.linspace(min_height,max_height,samples)
	plt.plot(x_norm,female_n*width*norm.pdf(x_norm,female_mean,female_std),color='r',label='Female Normal')
	plt.plot(x_norm,male_n*width*norm.pdf(x_norm,male_mean,male_std),color='b',label='Male Normal')

	#Bar Graph:
	x = np.asfarray(range(num_bins))
	x_ticks = np.around(bin_center(x,num_bins,min_height,max_height),1)
	plt.bar(x_ticks, male_bin_array, color='b', width=bar_width, align='center', alpha=0.5, label='Male')
	plt.bar(x_ticks, female_bin_array, color='r', width=bar_width, align='center', alpha=0.5, label='Female')
	plt.xlabel("Height (in)")
	plt.ylabel("Count")
	plt.title("Height distribution")
	plt.legend()
	plt.show()


head_num = 50;

#data_frame = pd.read_csv('Height.csv')
data_frame = pd.read_excel('Assignment_2_Data_and_Template.xlsx',sheetname='Data')
print data_frame.query('Sex=="Female"')

#data_frame["Total_Inches"] = data_frame["Height_Feet"]*12 + data_frame["Height_Inches"]

#print data_frame

#min_height = data_frame["Total_Inches"].min()
#max_height = data_frame["Total_Inches"].max()




