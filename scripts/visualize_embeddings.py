"""Project matched detection embeddings with PCA (not claimed paper t-SNE)."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_yolo.model import load_checkpoint
from spectral_yolo.data import detection_loader,move_detection
from spectral_yolo.loss import DetectionLoss


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    model,saved=load_checkpoint(args.checkpoint,args.device);model.eval()
    if not model.head.use_prototypes: raise ValueError('Prototype visualization requires cp_head=true')
    vectors,labels=[],[]
    # Sample embeddings at ground-truth box centers on P3, a transparent
    # visualization protocol distinct from assignment-based metric evaluation.
    for batch in detection_loader(args.data,model.config,args.split,1):
        batch=move_detection(batch,args.device)
        feature=model.features(batch['img'])[0]
        embedding=model.head.embedding[0](feature)
        for bbox,label in zip(batch['bboxes'],batch['cls'].flatten()):
            x=min(int(bbox[0]*embedding.shape[-1]),embedding.shape[-1]-1)
            y=min(int(bbox[1]*embedding.shape[-2]),embedding.shape[-2]-1)
            vectors.append(embedding[0,:,y,x].cpu())
            labels.append(int(label))
            if len(labels)>=args.max_samples: break
        if len(labels)>=args.max_samples: break
    if len(vectors)<2: raise ValueError('Need at least two annotated objects')
    matrix=torch.nn.functional.normalize(torch.stack(vectors),dim=-1).numpy()
    mean=matrix.mean(0);_,_,vh=np.linalg.svd(matrix-mean,full_matrices=False)
    xy=(matrix-mean)@vh[:2].T
    if xy.shape[1]<2: xy=np.pad(xy,((0,0),(0,2-xy.shape[1])))
    fig,ax=plt.subplots(figsize=(7,5),dpi=160)
    for c,name in enumerate(model.config['classes']):
        mask=np.array(labels)==c
        if mask.any(): ax.scatter(xy[mask,0],xy[mask,1],s=20,label=name,alpha=.7)
    title = 'SYNTHETIC geometric data - software diagnostic' if 'synthetic' in str(args.data).lower() else 'Reimplementation embeddings: supplied checkpoint/data'
    ax.set(xlabel='PC1',ylabel='PC2',title=title)
    ax.legend();fig.tight_layout()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out);plt.close(fig)
    np.savez(out.with_suffix('.npz'),embeddings=matrix,labels=np.array(labels),pca=xy)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--split',default='val');p.add_argument('--device',default='cpu')
    p.add_argument('--max-samples',type=int,default=1000);p.add_argument('--out',default='runs/embeddings.png')
    main(p.parse_args())
