from .gan_ablation import launch
SPEC=dict(name='vanilla_linear',method='vanilla',residual_g=True,fourier=True,g_fourier=True,latent=2,lr_g=.001,anneal=True,cosine=True,cosine_after=.5,alpha=.9,power=1)
if __name__=='__main__':launch('gan-ablation-vanilla-linear-v1',[SPEC],steps=50000,seeds=(0,1,2,3,4),workers=5,eval_samples=10000)
