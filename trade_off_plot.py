from adjustText import adjust_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set()
# sns.set_style('whitegrid')
# sns.set_palette('gist_yarg')
# sns.set_palette('pastel') #S12
sns.set_palette('Set2') #DOLP
# sns.set_palette('flare') #AD

rgb_psnr =  [33.92, 35.74, 36.98, 37.79, 
            #  40.56, 43.75, 44.47, 44.74]
             40.56498241, 43.76370633, 44.49306376, 44.73398534]
dolp_psnr = [25.43, 17.68, 16.69, 15.98,
            #  26.10, 26.75, 26.45, 25.20]
             27.21884116, 27.40629609, 26.4828493, 25.1910617403699]
 
# 散布図を描画
# plt.scatter(rgb_psnr, dolp_psnr)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(dolp_psnr, rgb_psnr)
# texts = [ax.text(dolp_psnr[i], rgb_psnr[i], text[i], ha='center', va='center') for i in range(len(dolp_psnr))]
# adjust_text(texts)
ax.set_ylabel('RGB PSNR [dB]')
ax.set_xlabel('DoLP PSNR [dB]')
# plt.show()
fig.savefig("trade-off.png")