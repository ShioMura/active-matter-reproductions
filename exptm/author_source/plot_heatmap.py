"""Plot saved control-grid values without interpolation or additional sampling."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);args=p.parse_args()
    root=args.results
    a=np.load(root/'heatmap.npz');m,s,t,ctrl=[a[k] for k in ['mean','sem','T','p']]
    assert m.shape == s.shape == (len(t),len(ctrl)) and np.isfinite(m).all()
    fig,ax=plt.subplots(figsize=(10,7))
    im=ax.imshow(m,origin='lower',aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='viridis')
    ax.set(xticks=range(len(ctrl)),xticklabels=[f'{v:g}' for v in ctrl],
           yticks=range(len(t)),yticklabels=[f'{v:g}' for v in t],
           xlabel='p control (prior mean)',ylabel='T control (prior variance)',
           title=f'ExpTM public checkpoint: {len(t)} × {len(ctrl)} control grid')
    #for i in range(len(t)):
    #    for j in range(len(ctrl)):
    #       ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',color='black' if m[i,j]>.55 else 'white',fontsize=8)
    fig.colorbar(im,ax=ax,label='Mean occupied fraction')
    fig.text(.5,.025,'20 × 20 lattice · 16 samples per cell · Fixed public checkpoint\nCode controls only; physical mapping and Figure 2 checkpoint identity unverified.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(root/'density_heatmap.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    colors=plt.cm.plasma(np.linspace(.05,.85,len(t)))
    for i,temp in enumerate(t):
        ax.errorbar(ctrl,m[i],yerr=s[i],marker='o',capsize=3,color=colors[i],label=f'T control = {temp:g}')
    ax.set(xlabel='p control (prior mean)',ylabel='Mean occupied fraction',ylim=(0,1),
        title='Density slices through the same control grid')
    ax.grid(alpha=.2);ax.legend(fontsize=9,loc='center left',bbox_to_anchor=(1.02,.5))
    fig.text(.5,.025,'Bars: ±1 sampling SEM for a fixed model, n = 16.\nShared random seed across cells; bars do not quantify training or physical-model error.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.75,bottom=.22,top=.91)
    fig.savefig(root/'density_slices.png',dpi=180);plt.close(fig)
    rows=json.loads((root/'results.json').read_text())
    import csv
    with (root/'density_grid.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
