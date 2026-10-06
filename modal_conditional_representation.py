"""Evaluate existing conditional CIFAR models without training."""
import modal
from pathlib import Path
ROOT = Path(__file__).resolve().parent
data_volume = modal.Volume.from_name('paired-discriminator-cifar10-data')
run_volume = modal.Volume.from_name('paired-discriminator-cifar10-runs')
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(uv_project_dir=str(ROOT), extras=['cifar'])
         .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TORCH_HOME': '/data/torch', 'PYTHONPATH': '/root'})
         .add_local_dir(ROOT / 'src' / 'paired_discriminator', '/root/paired_discriminator'))
app = modal.App('paired-discriminator-cifar-conditional-representation')

@app.function(image=image, gpu=['A10', 'L4', 'A100', 'H100', 'T4'], cpu=4, memory=16384,
              volumes={'/data': data_volume, '/runs': run_volume}, timeout=3600)
def evaluate():
    from paired_discriminator.cifar_conditional_representation import evaluate as run
    run_volume.reload()
    result = run('/runs/cifar10-integrity-v1', '/data/cifar10', '/runs/cifar10-conditional-representation-v1')
    data_volume.commit(); run_volume.commit()
    return {'evaluated_records': len(result)}

@app.local_entrypoint()
def main():
    print(evaluate.remote())
