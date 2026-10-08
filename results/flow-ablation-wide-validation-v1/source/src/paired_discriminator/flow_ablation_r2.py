from .flow_ablation import launch
SPECS=[
 dict(name='reference',depth=3,precondition=True,fourier=True),
 dict(name='cosine',depth=3,precondition=True,fourier=True,cosine=True),
 dict(name='batch1024',depth=3,precondition=True,fourier=True,batch=1024,cosine=True),
 dict(name='two_layers',depth=2,precondition=True,fourier=True,cosine=True),
 dict(name='late_time',depth=3,precondition=True,fourier=True,late=True,cosine=True),
 dict(name='wide256',width=256,depth=3,precondition=True,fourier=True,cosine=True),
]
if __name__=='__main__':launch('flow-ablation-r2-v1',SPECS,steps=50000,workers=6,eval_samples=10000)
