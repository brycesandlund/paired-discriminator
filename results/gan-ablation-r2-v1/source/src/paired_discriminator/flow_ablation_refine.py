from .flow_ablation import launch
SPECS=[dict(name='refine',depth=3,precondition=True,fourier=True,cosine=True,lr=.0001,resume=True)]
if __name__=='__main__':launch('flow-ablation-refine-v1',SPECS,steps=100000,seeds=(0,1,2,3,4),workers=5,eval_samples=10000)
