"""Run the complete software-only demonstration using synthetic data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from make_synthetic_data import generate


def main(args):
    root=Path(__file__).resolve().parents[1]
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    data=out/'synthetic_data';generate(data)
    commands=[
        ['scripts/smoke_test.py'],
        ['-m','spectral_yolo.train','--config','configs/smoke.yaml','--data',str(data/'data.yaml'),'--out',str(out/'detect')],
        ['-m','spectral_yolo.train','--config','configs/smoke.yaml','--task','joint','--manifest',str(data/'temporal.jsonl'),'--out',str(out/'joint')],
        ['-m','spectral_yolo.evaluate','--checkpoint',str(out/'detect/last.pt'),'--data',str(data/'data.yaml'),'--out',str(out/'detection_eval.json')],
        ['-m','spectral_yolo.evaluate','--task','forecast','--checkpoint',str(out/'joint/last.pt'),'--manifest',str(data/'temporal.jsonl'),'--out',str(out/'forecast_eval.json')],
        ['-m','spectral_yolo.predict','--checkpoint',str(out/'detect/last.pt'),'--image',str(data/'images/test/000.png'),'--out',str(out/'prediction.png')],
    ]
    for index,command in enumerate(commands):
        print('RUN',index+1, ' '.join(command),flush=True)
        result=subprocess.run([sys.executable,*command],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        (out/f'step_{index+1}.log').write_text(result.stdout,encoding='utf-8')
        if result.returncode:
            print(result.stdout);raise RuntimeError(f'Demo step {index+1} failed; see step log')
    report={'status':'PASS','scope':'SYNTHETIC SOFTWARE DEMO ONLY - NOT PAPER EXPERIMENTS',
            'steps':len(commands),'detection':json.loads((out/'detection_eval.json').read_text()),
            'forecast':json.loads((out/'forecast_eval.json').read_text()),'prediction_image':str(out/'prediction.png')}
    (out/'demo_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='runs/synthetic_demo')
    main(p.parse_args())
