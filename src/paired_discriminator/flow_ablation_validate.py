from .flow_ablation import launch
SPECS=[dict(name='cosine',depth=3,precondition=True,fourier=True,cosine=True),dict(name='two_layers',depth=2,precondition=True,fourier=True,cosine=True)]
if __name__=='__main__':launch('flow-ablation-validation-v1',SPECS,steps=50000,seeds=(1,2,3,4),workers=8,eval_samples=10000)
