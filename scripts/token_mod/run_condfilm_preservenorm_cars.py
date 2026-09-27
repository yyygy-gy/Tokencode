
import os, sys, subprocess, datetime
from pathlib import Path
root = Path(r'E:/Project/Hicropl_3/HiCroPL')
os.chdir(root)
os.environ['PYTHONPATH'] = str(root)
os.environ['PYTHONUNBUFFERED'] = '1'
data = r'E:/Project/dataset'
seed='1'; shots='16'; max_epoch='40'; cfg='vit_b16_ep50_k16'
trainer='TokenModHiCroPL'; dataset='stanford_cars'; alias='cars'
run_tag=f'seed{seed}_condfilm_preservenorm_e{max_epoch}'
train_dir=root/f'output/token_mod/train_base/{dataset}/shots_{shots}/{trainer}/{cfg}/{run_tag}'
test_dir=root/f'output/token_mod/test_new/{dataset}/shots_{shots}/{trainer}/{cfg}/{run_tag}'
log_dir=root/'output/token_mod/logs'
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
train_log=log_dir/f'{alias}_{run_tag}.out.txt'
test_log=log_dir/f'{alias}_{run_tag}_novel.out.txt'
status=log_dir/f'{alias}_{run_tag}.status.txt'
common=[
 'DATASET.NUM_SHOTS', shots,
 'OPTIM.MAX_EPOCH', max_epoch,
 'TRAIN.PRINT_FREQ', '20',
 'DATALOADER.TEST.BATCH_SIZE', '32',
 'DATALOADER.NUM_WORKERS', '0',
 'TRAINER.TOKENMOD.MODULATION', 'film',
 'TRAINER.TOKENMOD.USE_RESIDUAL', 'True',
 'TRAINER.TOKENMOD.USE_DISTILL', 'False',
 'TRAINER.TOKENMOD.RESIDUAL_GATE', 'False',
 'TRAINER.TOKENMOD.USE_CONDITIONAL_CODES', 'True',
 'TRAINER.TOKENMOD.VISUAL_CODE_SOURCE', 'image',
 'TRAINER.TOKENMOD.FILM_PRESERVE_NORM', 'True',
]
def now():
    return datetime.datetime.now().isoformat()
def run(cmd, log_path, phase):
    status.write_text(f'{phase} {now()}\n', encoding='utf-8')
    print(phase, flush=True)
    print('CMD:', ' '.join(cmd), flush=True)
    with open(log_path, 'w', encoding='utf-8', errors='replace') as f:
        p=subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(root))
        code=p.wait()
    if code!=0:
        status.write_text(f'FAIL {phase} exit={code} {now()}\n', encoding='utf-8')
        raise SystemExit(code)
train_cmd=[
 sys.executable,'-u','train.py',
 '--root', data,
 '--seed', seed,
 '--trainer', trainer,
 '--dataset-config-file', f'configs/datasets/{dataset}.yaml',
 '--config-file', f'configs/trainers/{trainer}/{cfg}.yaml',
 '--output-dir', str(train_dir),
 'DATASET.SUBSAMPLE_CLASSES','base',
]+common
test_cmd=[
 sys.executable,'-u','train.py',
 '--root', data,
 '--seed', seed,
 '--trainer', trainer,
 '--dataset-config-file', f'configs/datasets/{dataset}.yaml',
 '--config-file', f'configs/trainers/{trainer}/{cfg}.yaml',
 '--output-dir', str(test_dir),
 '--model-dir', str(train_dir),
 '--eval-only',
 'DATASET.SUBSAMPLE_CLASSES','new',
]+common
run(train_cmd, train_log, 'START train')
run(test_cmd, test_log, 'DONE train / START novel')
status.write_text(f'DONE all {now()}\n', encoding='utf-8')
print('DONE all', flush=True)
