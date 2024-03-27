from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import sys, glob
import corner


##
##	Run like:
##		projectScans_xy.py
##
##	Update 'chip', 'files', 'lables' with files of interest


######### Define global variables ###########
chip = "W02S09"
files = ["2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_40V", "2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_100V", "2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_180V", "2D_output_Astropix_W2_S9_pix0_00_2D_xyscan_230V"]
labels = ["40V", "100V", "180V", "230V"]

#chip = "W10S02"
#files = ["2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_2V", "2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_4V", "2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_8V", "2D_output_Astropix_W10_S9_pix0_01_2D_xyscan_10V"]
#labels = ["2V", "4V", "8V", "10V"]

profile_dfs=[]
grid_arrs=[]
x_arr=[]
y_arr=[]
x_arr_core=[]
y_arr_core=[]


savePlot = True
average = False #plot average peak value for profile ../plots. If False, plot total
core = True #plot core region in profile plots as well as total
out_str = '_core' if core else ''

######### import files ###########
for i,f in enumerate(files):
	df1=pd.read_csv(f'../../Laser_root_files/out/{f}.csv')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	img = ax.scatter(df1['x'], df1['y'], c=df1['max[mV]'], cmap=plt.hot()) #raw data, scale=1
	fig.colorbar(img)
	plt.xlabel('x [um] - toward PCB')
	plt.ylabel('y[um] - along pixel')
	fig.tight_layout()
	if savePlot:
		plt.savefig(f"../plots/sandbox_xy_{chip}/xy_{labels[i]}.png")
		plt.savefig(f"../plots/sandbox_xy_{chip}/xy_{labels[i]}.pdf")
		plt.clf()
	else:
		plt.show()

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
	
	####### plot 2D grid ###################
	data_arr = np.zeros((len(x_step), len(y_step)))
	for a,x in enumerate(x_step):
		for b,y in enumerate(y_step):
			row = df1.loc[(df1['x'] == x) & (df1['y'] == y)]
			data_arr[a,b] = row['max[mV]']
	grid_arrs.append(data_arr)	

	####### define profile hist data sets ###################
	sum_rows, sum_cols = data_arr.sum(axis=0), data_arr.sum(axis=1)
	if average:
		sum_rows = sum_rows/len(sum_rows)
		sum_cols = sum_cols/len(sum_cols)
	x_arr.append(sum_cols)
	y_arr.append(sum_rows)
	
	if core:
		max_pixel = data_arr.max()
		core_arr = data_arr.copy()
		core_arr[core_arr<max_pixel/2.] = np.nan
		sum_rows_core, sum_cols_core = np.nansum(core_arr, axis=0), np.nansum(core_arr, axis=1)
		if average:
			sum_rows_core = sum_rows_core/np.count_nonzero(~np.isnan(sum_rows_core))
			sum_cols_core = sum_cols_core/np.count_nonzero(~np.isnan(sum_rows_core))
		x_arr_core.append(sum_cols_core)
		y_arr_core.append(sum_rows_core)

	#definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02
	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	#shape figure
	plt.figure(1, figsize=(8, 8))
	axScatter = plt.axes(rect_scatter, label=f'forMatplotlib{labels[i]}')
	axHistx = plt.axes(rect_histx, label=f'forMatplotlib{labels[i]}')
	axHisty = plt.axes(rect_histy, label=f'forMatplotlib{labels[i]}')
	#remove labels
	nullfmt = NullFormatter()		  
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)
	#add axis labels
	axScatter.set_xlabel("x location [um]")
	axScatter.set_ylabel("y location [um]")
	if average:
		axHistx.set_ylabel("Average energy deposit [mV]")
		axHisty.set_xlabel("Average energy deposit [mV]")
	else:
		axHistx.set_ylabel("Total energy deposit [mV]")
		axHisty.set_xlabel("Total energy deposit [mV]")
	#plot
	axScatter.scatter(df1['x'], df1['y'], c=df1['max[mV]'], cmap=plt.hot())
	p1 = axHistx.plot(x_step, sum_cols, ds="steps")
	axHisty.plot(sum_rows, y_step, ds="steps")
	if core:
		p2 = axHistx.plot(x_step, sum_cols_core, ds="steps")
		axHisty.plot(sum_rows_core, y_step, ds="steps")
	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())
	plt.legend([p1, p2], labels=["full array", "core"], loc=[0.27, 1.2])#"upper right") 
	#save
	if savePlot:
		plt.savefig(f"../plots/sandbox_xy_{chip}/xy_arrs_{labels[i]}{out_str}.png")
		plt.savefig(f"../plots/sandbox_xy_{chip}/xy_arrs_{labels[i]}{out_str}.pdf")
		plt.clf()
	else:
		plt.show()

####### plotting for profile #########
fig = plt.figure()
ax = fig.add_subplot(111)
for j,df in enumerate(profile_dfs):
	img = ax.plot(df['x'], df['max[mV]'], ds='steps', label=labels[j])
plt.xlabel('x [um] - toward PCB')
plt.ylabel(f'analog pulse height [mV]')
plt.title(f'Profile at y={profile_y}')
plt.legend(loc='best')
fig.tight_layout()
if savePlot:
	plt.savefig(f"../plots/sandbox_xy_{chip}/y_profile.png")
	plt.savefig(f"../plots/sandbox_xy_{chip}/y_profile.pdf")
	plt.clf()
else:
	plt.show()
	
####### plotting for sums #########
plt_name_str = "average" if average else "total"
fig = plt.figure()
ax = fig.add_subplot(111)
for k,x in enumerate(x_arr):
	img = ax.plot(x_step, x, ds='steps', label=labels[k])
if core:
	plt.gca().set_prop_cycle(None) #reset colors
	for k,x in enumerate(x_arr_core):
		img2 = ax.plot(x_step, x, ds='steps', linestyle='dashed', label=labels[k]+"_core")
plt.xlabel('x [um] - toward PCB')
plt.ylabel(f'{plt_name_str} analog pulse height [mV]')
plt.title(f'{plt_name_str} amplitude at each x')
plt.legend(loc='best')
fig.tight_layout()
if savePlot:
	plt.savefig(f"../plots/sandbox_xy_{chip}/x_{plt_name_str}{out_str}.png")
	plt.savefig(f"../plots/sandbox_xy_{chip}/x_{plt_name_str}{out_str}.pdf")
	plt.clf()
else:
	plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
for m,y in enumerate(y_arr):
	img = ax.plot(y_step, y, ds='steps', label=labels[m])
if core:
	plt.gca().set_prop_cycle(None) #reset colors
	for m,y in enumerate(y_arr_core):
		img2 = ax.plot(y_step, y, ds='steps', linestyle='dashed', label=labels[m]+"_core")
plt.ylabel('y[um] - along pixel')
plt.ylabel(f'{plt_name_str} analog pulse height [mV]')
plt.title(f'{plt_name_str} amplitude at each y')
plt.legend(loc='best')
fig.tight_layout()
if savePlot:
	plt.savefig(f"../plots/sandbox_xy_{chip}/y_{plt_name_str}{out_str}.png")
	plt.savefig(f"../plots/sandbox_xy_{chip}/y_{plt_name_str}{out_str}.pdf")
	plt.clf()
else:
	plt.show()