import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('default')
plt.style.use('seaborn-muted')
# sns.set()
# sns.set_style('ticks')
x = np.linspace(1, 70, 100)
r = 1/x
y = 4 * (1-r)
z = r
c = [1.0]*100

# プロット
fig = plt.figure(figsize=(8,4))
plt.plot(r, y, label="Number of pixels of RGB image", linewidth = 3.0)
plt.plot(r, z, label="Number of pixels of Polar information", linewidth = 3.0)
# plt.plot(r, c, label="Conventional", linestyle=':', linewidth = 3.0)
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# 凡例の表示
# plt.legend()

# プロット表示(設定の反映)
plt.show()
fig.savefig("resolution.png")