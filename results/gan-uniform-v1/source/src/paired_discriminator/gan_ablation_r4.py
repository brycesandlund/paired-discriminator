from .gan_ablation import launch
BASE=dict(residual_g=True,fourier=True,g_fourier=True,latent=2,lr_g=.001,anneal=True,cosine=True,cosine_after=.5)
SPECS=[
 dict(BASE,name='strong_constant',cosine=False),
 dict(BASE,name='strong_latecos'),
 dict(BASE,name='uniform',alpha=0,power=1),
 dict(BASE,name='weak',alpha=.5,power=1),
 dict(BASE,name='linear',alpha=.9,power=1),
 dict(BASE,name='half_square',alpha=.5,power=2),
 dict(BASE,name='ema_fast',mass_decay=.9),
 dict(BASE,name='ema_slow',mass_decay=.999),
 dict(BASE,name='vanilla_strong',method='vanilla'),
]
if __name__=='__main__':launch('gan-ablation-r4-v1',SPECS,steps=50000,eval_samples=10000)
