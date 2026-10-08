from .gan_ablation import launch
SPECS=[
 dict(name='residual_raw',residual_g=True),
 dict(name='residual_fourier',residual_g=True,fourier=True),
 dict(name='residual_fast',residual_g=True,fourier=True,lr_g=.001),
 dict(name='residual_anneal',residual_g=True,fourier=True,lr_g=.001,anneal=True),
 dict(name='residual_fast_d',residual_g=True,fourier=True,lr_d=.001,anneal=True),
 dict(name='residual_fast_g',residual_g=True,fourier=True,lr_g=.001,lr_d=.0002,anneal=True),
 dict(name='residual_weak',residual_g=True,fourier=True,lr_g=.001,anneal=True,alpha=.5,power=1),
 dict(name='residual_uniform',residual_g=True,fourier=True,lr_g=.001,anneal=True,alpha=0,power=1),
]
if __name__=='__main__':launch('gan-ablation-r2-v1',SPECS)
