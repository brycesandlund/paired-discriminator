from .flow_ablation import launch
if __name__=='__main__':
    launch('flow-ablation-wide-validation-v1',[dict(name='wide256',width=256,depth=3,precondition=True,fourier=True,cosine=True)],steps=50000,seeds=(1,2,3,4),workers=4,eval_samples=10000)
