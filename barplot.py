import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
# sns.set_palette('gist_yarg')
# sns.set_palette('pastel') #S12
sns.set_palette('Set2') #DOLP
# sns.set_palette('flare') #AD

np.random.seed(2018)
df = pd.DataFrame({
    # 'FDKN': [0.00482, 0.00479, 0.00482, 0.00482, 0.00481], # S12 FDKN
    # 'DKN' : [0.00791, 0.0079, 0.00791, 0.00791, 0.00791], # S12 DKN
    # 'UNet': [0.00557, 0.00553, 0.0049, 0.00551, 0.0053], # S12 UNet
    # 'U2Net': [0.00484, 0.00478, 0.00481, 0.00468, 0.00462], # S12 U2Net
    # 'FDSR': [0.0048, 0.00479, 0.0048, 0.00483, 0.00482], # S12 FDSR
    # 'NLSPN': [0.00433, 0.00431, 0.00427, 0.00436, 0.00427], # S12 NLSPN Backbone
    # 'GuideNet': [0.00454, 0.00442, 0.00435, 0.00433, 0.00434], # S12 GuideNet Backbone
    # 'ENet' : [0.00431, 0.00428, 0.00428, 0.0043, 0.0043], # S12 ENet
    # 'Ours' : [0.00429, 0.0043, 0.00428, 0.00432, 0.00428], # S12 Ours
    # 'Ours (Opt)' : [0.00413, 0.00412, 0.00411, 0.00414, 0.00413], # S12 Ours Optimized

    # 'UNet': [0.00557, 0.00553, 0.0049, 0.00551, 0.0053], # S12 UNet
    # 'Ours': [0.00432, 0.00434, 0.00437, 0.00432, 0.00435], # S12 One Branch L2
    # 'One Branch': [0.00432, 0.00434, 0.00437, 0.00432, 0.00435], # S12 One Branch L2
    # 'Two Branch': [0.00427, 0.00426, 0.00429, 0.00431, 0.00429], # S12 Two Branch L2
    # 'w/ Conf': [0.00427, 0.00426, 0.00429, 0.00431, 0.00429], # S12 Two Branch L2
    # 'w/o Conf': [0.00421, 0.00419, 0.00421, 0.0042, 0.00419], # S12 Two Branch L2 NoConf
    # 'Baseline': [0.00433, 0.00436, 0.00435, 0.00437, 0.00439], # S12 Two Branch L1
    # 'w/ Mask': [0.00437, 0.00438, 0.00432, 0.00436, 0.00433], # S12 Two Branch L1 w/ Mask
    # 'w/ Sparse S0': [0.00434, 0.00432, 0.00432, 0.00432, 0.00433], # S12 Two Branch L1 w/ sps s0
    # 'w/ Sparse S0 + Mask': [0.00429, 0.0043, 0.00428, 0.00432, 0.00428], # S12 Two Branch L1 w/ sps s0 + mask
    # 'w/ Sparse S0 + Mask': [0.00434, 0.00431, 0.0043, 0.00429, 0.0043], # S12 Two Branch L1 w/ sps s0 + mask (2x2)
    # 'L2 Loss': [0.00428, 0.00432, 0.00431, 0.00423, 0.00426], # S12 Two Branch L2 w/ sps s0 + mask
    # 'L1 Loss': [0.00429, 0.0043, 0.00428, 0.00432, 0.00428], # S12 Two Branch L1 w/ sps s0 + mask
    # 'Sparse S12 GT & w/ Sparse S0 + Mask': [0.00406, 0.00404, 0.00408, 0.00408, 0.00406] # S12 Two Branch L1 w/ sps s0 + mask (2x2) sparse s12 GT (Ramdom)
    # 'Sparse S12 GT & w/ Sparse S0 + Mask': [0.00409, 0.00406, 0.00407, 0.00405, 0.00405] # S12 Two Branch L1 w/ sps s0 + mask (2x2) sparse s12 GT

    # 'FDKN': [27.456, 27.565, 27.623, 27.641, 27.176], # DoLP FDKN
    # 'DKN' : [27.792, 27.782, 27.657, 27.852, 27.652], # DoLP DKN
    # 'UNet': [28.638, 28.761, 28.802, 28.703, 28.839], # DoLP UNet
    # 'U2Net': [28.777, 28.958, 28.733, 29.215, 29.19], # DoLP U2Net
    # 'FDSR': [29.414, 29.437, 29.406, 29.41, 29.422], # DoLP FDSR
    # 'GuideNet': [29.697, 29.683, 29.661, 29.817, 29.688], # DoLP GuideNet Backbone
    # 'PENet' : [29.633, 29.782, 29.753, 29.735, 29.719], # DoLP ENet
    # 'NLSPN': [29.778, 29.795, 29.964, 29.699, 29.905], # DoLP NLSPN Backbone
    'Ours' : [30.04, 30.041, 30.038, 29.937, 30.043], # DoLP Ours
    'Ours 2dec' : [30.30012817, 30.26014623, 30.19431756, 30.32957671, 30.33650757], # DoLP Ours 2dec
    # 'Ours (Opt)' : [30.274, 30.316, 30.312, 30.237, 30.267], # DoLP Ours Optimized

    # 'UNet': [28.638, 28.761, 28.802, 28.703, 28.839], # DoLP UNet
    # 'Ours': [29.559, 29.248, 29.58, 29.628, 29.597], # DOLP One Branch L2
    # 'One Branch': [29.559, 29.248, 29.58, 29.628, 29.597], # DOLP One Branch L2
    # 'Two Branch': [29.812, 29.743, 29.656, 29.786, 29.826], # DOLP Two Branch L2
    # 'w/ Conf': [29.812, 29.743, 29.656, 29.786, 29.826], # DOLP Two Branch L2
    # 'w/o Conf': [29.864, 30.036, 30.027, 29.835, 29.99], # DOLP Two Branch L2 No conf
    # 'Baseline': [29.823, 29.914, 29.994, 29.935, 29.935], # DOLP Two Branch L1
    # 'w/ Mask': [29.877, 29.917, 30.041, 29.914, 30.026], # DOLP Two Branch L1 w/ Mask
    # 'w/ Sparse S0': [29.936, 29.956, 29.979, 29.944, 29.954], # DOLP Two Branch L1 w/ sps s0
    # 'w/ Sparse S0 + Mask': [30.04, 30.041, 30.038, 29.937, 30.043], # DOLP Two Branch L1 w/ sps s0 + mask
    # 'w/ Sparse S0 + Mask': [29.912, 30.033, 30.012, 29.993, 30.008], # DOLP Two Branch L1 w/ sps s0 + mask (2x2)
    # 'L2 Loss': [29.755, 29.657, 29.71, 29.916, 29.766], # DOLP Two Branch L1 w/ sps s0 + mask
    # 'L1 Loss': [30.04, 30.041, 30.038, 29.937, 30.043], # DOLP Two Branch L1 w/ sps s0 + mask
    # 'Sparse S12 GT & w/ Sparse S0 + Mask': [30.442, 30.471, 30.465, 30.393, 30.399] # DOLP Two Branch L1 w/ sps s0 + mask (2x2) sparse s12 GT (Random)
    # 'Sparse S12 GT & w/ Sparse S0 + Mask': [30.248, 30.449, 30.312, 30.394, 30.446] # DOLP Two Branch L1 w/ sps s0 + mask (2x2) sparse s12 GT

    # 'FDKN': [3.542, 3.519, 3.515, 3.509, 3.563], # AD FDKN
    # 'DKN' : [3.496, 3.495, 3.477, 3.499, 3.518], # AD DKN
    # 'UNet': [3.114, 2.998, 3.135, 2.905, 2.946], # AD UNet
    # 'U2Net': [2.804, 2.782, 2.926, 2.913, 2.726], # AD U2Net
    # 'FDSR': [2.839, 2.838, 2.842, 2.854, 2.844], # AD FDSR
    # 'GuideNet': [2.765, 2.743, 2.75, 2.705, 2.772], # AD GuideNet Backbone
    # 'PENet' : [2.787, 2.712, 2.74, 2.732, 2.734], # AD ENet
    # 'NLSPN': [2.678, 2.677, 2.64, 2.726, 2.655], # AD NLSPN Backbone
    # 'Ours' : [2.577, 2.578, 2.577, 2.595, 2.573], # AD Ours
    # 'Ours 2dec' : [2.568032172, 2.578416603, 2.574749898, 2.561272044, 2.585654774], # AD Ours 2dec
    # 'Ours (Opt)' : [2.524, 2.518, 2.517, 2.529, 2.522], # AD Ours Opt
    # 'Ours 2dec (Opt)' : [2.512489018, 2.531103343, 2.538572678, 2.511492336, 2.507939706], # AD Ours 2decOpt

    # 'UNet': [3.114, 2.998, 3.135, 2.905, 2.946], # AD UNet
    # 'Ours': [2.847, 3.032, 2.819, 2.797, 2.79], # AD One Branch
    # 'One Branch': [2.847, 3.032, 2.819, 2.797, 2.79], # AD One Branch
    # 'Two Branch': [2.7, 2.729, 2.761, 2.695, 2.697], # AD Two Branch
    # 'w/ Conf': [2.7, 2.729, 2.761, 2.695, 2.697], # AD Two Branch
    # 'w/o Conf': [2.748, 2.643, 2.642, 2.734, 2.673], # AD Two Branch No conf
    # 'Baseline': [2.596, 2.601, 2.588, 2.597, 2.605], # AD Two Branch L1
    # 'w/ Mask': [2.605, 2.6, 2.577, 2.6, 2.58], # AD Two Branch L1 w/ Mask
    # 'w/ Sparse S0': [2.596, 2.592, 2.585, 2.591, 2.591], # AD Two Branch L1 w/ sps s0
    # 'w/ Sparse S0 + Mask': [2.577, 2.578, 2.577, 2.595, 2.573], # AD Two Branch L1 w/ sps s0 + mask
    # 'w/ Sparse S0 + Mask': [2.605, 2.58, 2.587, 2.586, 2.584], # AD Two Branch L1 w/ sps s0 + mask (2x2)
    # 'L2 Loss': [2.742, 2.752, 2.748, 2.667, 2.694], # AD Two Branch L2 w/ sps s0 + mask
    # 'L1 Loss': [2.577, 2.578, 2.577, 2.595, 2.573], # AD Two Branch L1 w/ sps s0 + mask
    # 'Sparse S12 GT & w/ Sparse S0 + Mask': [2.485, 2.48, 2.487, 2.495, 2.489] # AD Two Branch L1 w/ sps s0 + mask (2x2) sparse s12 GT
    # 'Sparse S12 GT & w/ Sparse S0 + Mask': [2.511, 2.489, 2.5, 2.491, 2.487] # AD Two Branch L1 w/ sps s0 + mask (2x2) sparse s12 GT (Random)
})

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# bar chart & error bar
x_position = np.arange(len(df.columns))
# error_bar_set = dict(lw = 1, capthick = 1, capsize = 20)
error_bar_set = dict(lw = 1, capthick = 1, capsize = 10)
ax.bar(x_position, df.mean(), yerr=df.std(), tick_label=df.columns, error_kw=error_bar_set)

# jitter plot
df_melt = pd.melt(df)
print(df_melt.head())

sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax = ax, size=3)

ax.set_xlabel(' ')
ax.tick_params(labelsize=8)
# ax.tick_params(labelsize=12)

# ax.set_title('S12 RMSE ↓')
# plt.ylim(0.004, 0.006) #S12 RMSE
# plt.ylim(0.00420, 0.00434) #S12 RMSE
# plt.ylim(0.004, 0.0044) #S12 RMSE

ax.set_title('DoLP PSNR[dB] ↑')
# plt.ylim(28.5, 30.0) #DOLP PSNR
# plt.ylim(27.0, 30.5) #DOLP PSNR
# plt.ylim(29.8, 30.1) #DOLP PSNR
plt.ylim(29.9, 30.4) #DOLP PSNR

# ax.set_title('ADMap Angle ↓')
# plt.ylim(2.0, 3.5)
# plt.ylim(2.56, 2.60)
# plt.ylim(2.5, 3.6)
# plt.ylim(2.5, 3.6)

ax.set_ylabel(' ')

plt.show()

fig.savefig("barplot.png")