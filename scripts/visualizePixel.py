from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import sys

######## run options ##############

interpolate = False
savePlot = False

######### import file1 ###########

df1=pd.read_csv(sys.argv[1])
df2=pd.read_csv(sys.argv[2])

df1_max_row = df1['max[mV]'].idxmax()
print(f"Max. amplitude from xz scan = {df1['max[mV]'].iloc[df1_max_row]:.2f} (8% laser intensity) at ({df1['x'].iloc[df1_max_row]}, {df1['y'].iloc[df1_max_row]}, {df1['z'].iloc[df1_max_row]})")
df2_max_row = df2['max[mV]'].idxmax()
df2_max = df2['max[mV]'].iloc[df2_max_row]
print(f"Max. amplitude from xy scan = {df2_max:.2f} (35% laser intensity) at ({df2['x'].iloc[df2_max_row]}, {df2['y'].iloc[df2_max_row]}, {df2['z'].iloc[df2_max_row]})")


####interpolate between datapoints - xz scan only
x_interp = np.unique(df1['x'])
y_interp = np.unique(df1['y'])
z_interp = np.unique(df1['z'])
data_interp = np.zeros((len(x_interp), len(z_interp)))
for i,x in enumerate(x_interp):
	for k,z in enumerate(z_interp):
		row = df1.loc[(df1['x'] == x) & (df1['z'] == z)]
		data_interp[i,k] = row['max[mV]']

interp = RegularGridInterpolator((x_interp, z_interp), data_interp)

#define interpolated data on finer mesh
dx2, dy2, dz2 = 25, 1, 100
x2 = np.arange(x_interp.min(), x_interp.max(), dx2)
z2 = np.arange(z_interp.min(), z_interp.max(), dz2)
X2, Z2 = np.meshgrid(x2,z2)
dat=interp((X2, Z2))

if interpolate:
	#plot interpolation
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	nm="interp_linear"
	img = ax.scatter(X2, 1520., Z2, c=dat, cmap=plt.hot())
	#img = ax.scatter(df1['x'], df1['y'], df1['z'], c=df1['max[mV]'], cmap=plt.hot()) #raw data, scale=1
	fig.colorbar(img)
	plt.xlabel('x [um] - toward PCB')
	plt.ylabel('y[um] - along pixel')
	ax.set_zlabel('z[um] - into bulk')
	fig.tight_layout()
	if savePlots:
		plt.savefig("sandbox_100V_xz_W02S09/"+nm+".png")
		plt.savefig("sandbox_100V_xz_W02S09/"+nm+".pdf")
	else:
		plt.show()

####### compare files #######
print("Scale xy scan to match max of xz scan in intersecting line")
df2_1520 = df2[df2['y'].isin([1520.])]
df2_1520.reset_index(inplace=True)
df2_max_row_1520 = df2_1520['max[mV]'].idxmax()
maxXY = interp((df2_1520['x'].iloc[df2_max_row_1520], df2_1520['z'].iloc[df2_max_row_1520]))
print(f"Max. amplitude from xy scan in intersecting plane = {df2_1520['max[mV]'].iloc[df2_max_row_1520]:.2f} (35% laser intensity) at ({df2_1520['x'].iloc[df2_max_row_1520]}, {df2_1520['y'].iloc[df2_max_row_1520]}, {df2_1520['z'].iloc[df2_max_row_1520]})")
print(f"Value of xy max pixel from xy scan = {df2_1520['max[mV]'].iloc[df2_max_row_1520]:.2f}")
print(f"Value of xy max pixel from xz scan = {maxXY:.2f}")

scale = maxXY / df2_max
print(f"Assume linearity? Scale xy scan by {scale:.2f}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nm = 'xy_xz_scaled3.07'
x = pd.concat([df1['x'], df2['x']])
y = pd.concat([df1['y'], df2['y']])
z = pd.concat([df1['z'], df2['z']])
c = pd.concat([df1['max[mV]'],  df2['max[mV]'].apply(lambda x: x*scale)])

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.xlabel('x [um] - toward PCB')
plt.ylabel('y[um] - along pixel')
ax.set_zlabel('z[um] - into bulk')
fig.tight_layout()
if savePlot:
	plt.savefig("sandbox_100V_xz_W02S09/"+nm+".png")
	plt.savefig("sandbox_100V_xz_W02S09/"+nm+".pdf")
else:
	plt.show()