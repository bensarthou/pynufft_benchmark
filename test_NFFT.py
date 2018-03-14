import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim
import sys
from sympy.utilities.iterables import variations

def compare_mse(x, y):
	return np.linalg.norm(x - y)

argv = sys.argv[1:]
n = len(argv)

fig, axes = plt.subplots(int(n*(n-1)/2.),3)
axes = np.atleast_2d(axes)
# plt.suptitle('Comparaison: '+' VS '.join(argv))
list_time_pre, list_time_proc, list_title = [],[],[]
cnt = 0
for i in range(n):
	dict_1 = np.load('datas/'+argv[i])[()]
	y_1 = dict_1['y']
	list_time_pre.append(dict_1['time_preproc'])
	list_time_proc.append(dict_1['time_proc'])
	list_title.append(dict_1['title'])
	for j in range(i,n):
		if j!=i:
			# print(i,j)
			dict_2 = np.load('datas/'+argv[j])[()]
			y_2 = dict_2['y']
			y_1_abs, y_2_abs = np.absolute(y_1), np.absolute(y_2)

			mse = compare_mse(y_1_abs/np.max(y_1_abs), y_2_abs/np.max(y_2_abs))
			ssim = compare_ssim(y_1_abs, y_2_abs, data_range=y_1_abs.max() - y_1_abs.min())

			## Plot
			axes[cnt,0].plot(y_1_abs)
			axes[cnt,0].set_title(dict_1['title']+'\n MSE:'+str(mse), fontsize = 10)

			axes[cnt,1].plot(y_2_abs)
			axes[cnt,1].set_title(dict_2['title']+'\n SSIM:'+str(ssim), fontsize = 10)

			axes[cnt,2].set_title('K-space')
			axes[cnt,2].plot(np.real(y_1),np.imag(y_1), 'r.')
			axes[cnt,2].plot(np.real(y_2),np.imag(y_2), 'b.')
			cnt+=1

plt.tight_layout()
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.savefig('results/perf_comp_'+'_VS_'.join(list_title)+'.png', bbox_inches='tight')
plt.show()

## Plot time Comparaison
ind = np.arange(n)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, list_time_pre, width, color='#d62728')
p2 = plt.bar(ind, list_time_proc, width, bottom=list_time_pre)

plt.ylabel('Processing times')
plt.title('Performance times: '+ '/ '.join(list_title))
plt.xticks(ind, list_title)
plt.legend((p1[0], p2[0]), ('preproc', 'processing'))
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.savefig('results/time_comp_'+' VS '.join(list_title)+'.png',bbox_inches='tight')
plt.show()
