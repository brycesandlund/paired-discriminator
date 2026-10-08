from .gan_ablation import launch
BASE=dict(residual_g=True,fourier=True,lr_g=.001,anneal=True,latent=2,g_fourier=True)
SPECS=[
 dict(BASE,name='latent2',g_fourier=False),
 dict(BASE,name='g_fourier'),
 dict(BASE,name='g_fourier_noskip',residual_g=False),
 dict(BASE,name='g_fourier_z16',latent=16),
 dict(BASE,name='g_fourier_raw_d',fourier=False),
 dict(BASE,name='g_fourier_cosine',cosine=True),
 dict(BASE,name='g_fourier_silu_d',silu=True),
 dict(BASE,name='g_fourier_beta0',betas=[0.,.99]),
]
if __name__=='__main__':launch('gan-ablation-r3-v1',SPECS)
