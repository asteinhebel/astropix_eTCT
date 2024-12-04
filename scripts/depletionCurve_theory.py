import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

################################################
## 	To run:
##		depletionCurve_theory.py <waferResistivity:float or list of floats> <error:float(optional, percent)>
##
## 	If no 'error' option given, assume 30% uncertainty
################################################

def resToDopingConc(res):
	#from 'computePConcentration' (view-source:http://www.solecon.com/sra/rho2ccal.htm), back end of Java calculator (http://www.solecon.com/sra/rho2ccal.htm)
	r = np.log(res)
	pMu = 482.8 / (1 + 0.1322 / pow(res, 0.811))
	conc = np.exp(43.28 - r - np.log(pMu))
	return conc

def eqn_depletion(res, bias):
	d = np.sqrt(2*SI_PERM*bias / ELEM_CHARGE / resToDopingConc(res)) #in cm
	return d*10000. #returns um
	

################# main ##############################

### global variables, constants ###

SI_PERM = 1.03e-12
ELEM_CHARGE = 1.60e-19
BIAS_RANGE = np.arange(0, 500, 20)

#change font size on plot
plt.rcParams.update({'font.size': 18})

### input parameters ###
try:
	res = [float(sys.argv[1])]
except ValueError:
	if ',' in sys.argv[1]:
		res_in = sys.argv[1].split(',')
		res = [float(r) for r in res_in]
	else:
		print(f"{sys.argv[1]} cannot be typecast to a float. Try again")
		sys.exit()

try:
	errPerc = [float(sys.argv[2])/100.]
except ValueError:
	if ',' in sys.argv[2]:
		err_in = sys.argv[2].split(',')
		errPerc = [float(e)/100. for e in err_in]
	else:
		print(f"{sys.argv[2]} cannot be typecast to a float. Try again")
		sys.exit()
except IndexError: #no value given
	errPerc = [0.3]*len(res)
	
#err = res*errPerc
err = [a*b for a,b in zip(res,errPerc)]
print(f"Considering a wafer of {res}+\-{err} Ohm*cm")


### calculate depletion depth as a function of bias  ###
plt.figure(figsize=(10,6))
for i,r in enumerate(res):
	d_low = [eqn_depletion(r-err[i],v) for v in BIAS_RANGE]
	d_high = [eqn_depletion(r+err[i],v) for v in BIAS_RANGE]
	plt.fill_between(BIAS_RANGE, d_low, d_high, alpha=0.7, label=f"{int(r)}+\-{err[i]:.3} Ohm*cm")

ax = plt.gca()
plt.legend(loc="best")
plt.xlabel('Bias voltage [V]')
plt.ylabel('Depletion depth [um]')
ax.set_ylim([0, 500])
plt.tight_layout()
plt.savefig('theory_depletion.png')
plt.show()
