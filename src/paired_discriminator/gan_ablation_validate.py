from .gan_ablation import launch
BASE=dict(residual_g=True,fourier=True,g_fourier=True,latent=2,lr_g=.001,anneal=True,cosine=True,cosine_after=.5)
SPECS=[dict(BASE,name='ema_fast',mass_decay=.9,seeds=[1,2,3,4]),dict(BASE,name='linear',alpha=.9,power=1,seeds=[1,2,3,4]),dict(BASE,name='vanilla_fast',mass_decay=.9,method='vanilla',seeds=[0,1,2,3,4])]
if __name__=='__main__':launch('gan-ablation-validation-v1',SPECS,steps=50000,eval_samples=10000)
