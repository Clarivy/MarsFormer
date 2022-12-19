# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from performer_pytorch import SelfAttention

# %%
d = 64
Nh = 4
m = 256
m_s = [16,32,64,128,192,256,384,512,1024,2048]
n_samp = 5

# %%
attn1 = nn.MultiheadAttention(d, Nh, dropout=0.1, batch_first=False).cuda()

x = torch.randn(1, 1024, 64).cuda()
xt = x.transpose(0,1)
y1,yw = attn1(query=xt, key=xt, value=xt) # (1, 1024, 512)
y1 = y1.transpose(0,1)

# %%
MARE = []
MSE = []
for m in m_s:
    MARE_tmp = []
    MSE_tmp = []
    for i in range(n_samp):
        attn2 = SelfAttention(
            dim = d,
            heads = Nh,
            dim_head=64//4,
            dropout=0.1,
            #nb_features = m,
            causal = True,
        ).cuda()

        attn2.to_q.weight = nn.Parameter(attn1.in_proj_weight[:d,:])
        attn2.to_k.weight = nn.Parameter(attn1.in_proj_weight[d:2*d,:])
        attn2.to_v.weight = nn.Parameter(attn1.in_proj_weight[2*d:3*d,:])

        attn2.to_q.bias = nn.Parameter(attn1.in_proj_bias[:d])
        attn2.to_k.bias = nn.Parameter(attn1.in_proj_bias[d:2*d])
        attn2.to_v.bias = nn.Parameter(attn1.in_proj_bias[2*d:3*d])

        attn2.to_out.weight = nn.Parameter(attn1.out_proj.weight)
        attn2.to_out.bias = attn1.out_proj.bias

        y2 = attn2(x) # (1, 1024, 512)

        MARE_tmp.append(torch.mean(torch.abs((y1-y2)/y1))) # mean absolute relative error
        MSE_tmp.append(torch.mean((y1-y2)**2)) # mean squared error
    MARE.append(MARE_tmp)
    MSE.append(MSE_tmp)

MARE = np.array(MARE)
MSE = np.array(MSE)

# %%
import matplotlib.pyplot as plt


# %%
fig,axs = plt.subplots(1,2,figsize=(8,3))
plt.subplots_adjust(wspace=0.4)
ax = axs[0]
ax.plot(m_s,MARE,'o')
ax.set_title('FAVOR+ attention vs vanilla attention')
ax.set_ylabel('Mean absolute relative error')
ax.set_xlabel('# features m')


ax = axs[1]
ax.plot(m_s,MSE,'o')
ax.set_title('FAVOR+ attention vs vanilla attention')
ax.set_ylabel('Mean squared error')
ax.set_xlabel('# features m')

# %%
are = torch.abs((y1-y2)/y1)

# %%
fig,ax = plt.subplots(figsize=(3,3))
ax.plot(np.log10(np.abs(y1.view(-1).cpu().detach())),np.log10(are.view(-1).cpu().detach()),'.')
ax.set_xlabel('log10 abs(attention output)')
ax.set_ylabel('log10 abs relative error')

# %%
import seaborn as sns

# %%
fig,ax = plt.subplots(figsize=(3,3))
sns.histplot(ax=ax, x=np.log10(np.abs(y1.view(-1).cpu().detach())), y=np.log10(are.view(-1).cpu().detach()))
ax.set_xlabel('log10 abs(attention output)')
ax.set_ylabel('log10 abs relative error')
xl = ax.get_xlim()
ax.plot(xl,[0,0],'k:')
ax.plot(xl,[-1,-1],'k:')
ax.set_xlim(xl)

# %%



