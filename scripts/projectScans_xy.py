from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import sys, glob
import corner

######### import file1 ###########
"""
dir = sys.argv[1]

profile=[]
for f in glob.glob("*.csv"):
	df = pd.read_csv(f)
	x, y, z = df['x'], df['y'], df['z']
	ampMax = df['max[mV]']
"""

chip = "W02S09"
files = ["2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_40V", "2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_100V", "2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_180V", "2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_230V"]
labels = ["40V", "100V", "180V", "230V"]

#chip = "W10S02"
#files = ["2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_2V", "2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_4V", "2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_8V", "2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_10V"]
#labels = ["2V", "4V", "8V", "10V"]

profile_dfs=[]
grid_arrs=[]
x_sum=[]
y_sum=[]

for i,f in enumerate(files):
	df1=pd.read_csv(f'../Laser_root_files/out/{f}.csv')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	img = ax.scatter(df1['x'], df1['y'], c=df1['max[mV]'], cmap=plt.hot()) #raw data, scale=1
	fig.colorbar(img)
	plt.xlabel('x [um] - toward PCB')
	plt.ylabel('y[um] - along pixel')
	fig.tight_layout()
	#plt.show()
	plt.savefig(f"sandbox_xy_{chip}/xy_{labels[i]}.png")
	plt.savefig(f"sandbox_xy_{chip}/xy_{labels[i]}.pdf")
	plt.clf()

	####### middle profile ###################
	#do only once and use same values consistently 
	if i==0:
		x_step = np.unique(df1['x'])
		y_step = np.unique(df1['y'])
		y_ind = int(len(y_step)/2)
		profile_y = y_step[y_ind]
	
	df_profile = df1[df1['y'].isin([profile_y])]
	df_profile.reset_index(inplace=True)
	profile_dfs.append(df_profile)
	
	####### x and y grid ###################
	data_arr = np.zeros((len(x_step), len(y_step)))
	for a,x in enumerate(x_step):
		for b,y in enumerate(y_step):
			row = df1.loc[(df1['x'] == x) & (df1['y'] == y)]
			data_arr[a,b] = row['max[mV]']
	grid_arrs.append(data_arr)	

	sum_rows, sum_cols = data_arr.sum(axis=0), data_arr.sum(axis=1)
	x_sum.append(sum_cols)
	y_sum.append(sum_rows)

	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02
	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	# shape figure
	plt.figure(1, figsize=(8, 8))
	axScatter = plt.axes(rect_scatter, label=f'forMatplotlib{labels[i]}')
	axHistx = plt.axes(rect_histx, label=f'forMatplotlib{labels[i]}')
	axHisty = plt.axes(rect_histy, label=f'forMatplotlib{labels[i]}')
	# remove labels
	nullfmt = NullFormatter()         
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)
	#plot
	axScatter.scatter(df1['x'], df1['y'], c=df1['max[mV]'], cmap=plt.hot())
	axHistx.plot(x_step, sum_cols, ds="steps")
	axHisty.plot(sum_rows, y_step, ds="steps")
	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())
	#save
	plt.savefig(f"sandbox_xy_{chip}/xy_sums_{labels[i]}.png")
	plt.savefig(f"sandbox_xy_{chip}/xy_sums_{labels[i]}.pdf")
	plt.clf()

####### plotting for profile #########
fig = plt.figure()
ax = fig.add_subplot(111)
for j,df in enumerate(profile_dfs):
	img = ax.plot(df['x'], df['max[mV]'], ds='steps', label=labels[j])
plt.xlabel('x [um] - toward PCB')
plt.ylabel('analog pulse height [mV]')
plt.title(f'Profile at y={profile_y}')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(f"sandbox_xy_{chip}/y_profile.png")
plt.savefig(f"sandbox_xy_{chip}/y_profile.pdf")

####### plotting for sums #########
fig = plt.figure()
ax = fig.add_subplot(111)
for k,x in enumerate(x_sum):
	img = ax.plot(x_step, x, ds='steps', label=labels[k])
plt.xlabel('x [um] - toward PCB')
plt.ylabel('total analog pulse height [mV]')
plt.title(f'Total amplitude at each x')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(f"sandbox_xy_{chip}/x_sum.png")
plt.savefig(f"sandbox_xy_{chip}/x_sum.pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
for m,y in enumerate(y_sum):
	img = ax.plot(y_step, y, ds='steps', label=labels[m])
plt.ylabel('y[um] - along pixel')
plt.ylabel('total analog pulse height [mV]')
plt.title(f'Total amplitude at each y')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(f"sandbox_xy_{chip}/y_sum.png")
plt.savefig(f"sandbox_xy_{chip}/y_sum.pdf")