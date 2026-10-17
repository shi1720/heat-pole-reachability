import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heatpoles import *
ROOT=Path(__file__).resolve().parents[1];R=json.loads((ROOT/'results/study.json').read_text())
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42,'axes.labelsize':11})
colors=['#1b4965','#d1495b','#00857d','#a46313']
p=Potential((.8,),(0.,));W=np.load(ROOT/'results/witness.npz');theta=W['theta']
def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf');fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7),gridspec_kw={'width_ratios':[1.2,1]})
x=np.linspace(0,2*np.pi,401);y=np.linspace(.01,.9,260);X,Y=np.meshgrid(x,y);f=p.value(X+1j*Y,.02)
C=np.array([axes(v,.02,1,180)[0] for v in y])[:,None];S=np.array([axes(v,.02,1,180)[1] for v in y])[:,None]
r=np.sqrt(f.real**2/C+f.imag**2/S)
im=ax[0].pcolormesh(x,y,np.log10(r),cmap='viridis',vmin=-5,vmax=0,rasterized=True,shading='auto')
ax[0].contour(X,Y,r,levels=[.01],colors=['white'],linewidths=1.3)
ax[0].plot(np.pi,np.arccosh(np.exp(.02)/.8),'wx',mew=1.5,ms=7,label='Nominal pole')
ax[0].plot(np.pi,.59,'o',color='#f4a261',ms=5,label='Constructed pole')
ax[0].axhline(.5816,color='white',lw=.8,ls=':')
ax[0].set(xlabel='$x$',ylabel='$y$',title='(a) Exact tail budget needed for a pole')
ax[0].legend(loc='lower left',fontsize=8,facecolor='#173a40',labelcolor='white',framealpha=.8)
fig.colorbar(im,ax=ax[0],label=r'$\log_{10} r_N(x+iy;\tau)$',pad=.02)
k=np.arange(2,181);a=theta[:179]/k
ax[1].stem(k[:44],abs(a[:44]),linefmt=colors[0],markerfmt='.',basefmt=' ')
ax[1].set(xlabel='Unresolved Fourier mode $k$',ylabel='$|a_k|$',title='(b) Explicit positive-data witness',xlim=(1,45))
ax[1].text(.96,.94,'$\|h\|_1=0.0085523$\n$\phi_0+h > 0.1960$',transform=ax[1].transAxes,ha='right',va='top',fontsize=10)
fig.tight_layout(w_pad=1.0);save(fig,'reachability')
fig,ax=plt.subplots(1,2,figsize=(7,2.6))
for color,eta in zip(colors,[.001,.01,.05,.1]):
 rows=[q for q in R['widths'] if q['eta']==eta]
 ts=[q['tau'] for q in rows];width=[q['width'] for q in rows]
 ax[0].plot(ts,width,'o-',color=color,label=f'$\eta={eta:g}$',ms=3)
 ax[1].plot(ts,[100*(1-q['width']/q['nominal']) for q in rows],'o-',color=color,ms=3)
nom=[q for q in R['widths'] if q['eta']==.01]
ax[0].plot([q['tau'] for q in nom],[q['nominal'] for q in nom],'k--',lw=1,label='Nominal')
ax[0].set(xscale='log',xlabel=r'$\tau=\nu t$',ylabel='Analyticity half-width',title='(a) Sharp robust strip widths')
ax[0].legend(fontsize=8,ncol=2)
ax[1].set(xscale='log',xlabel=r'$\tau=\nu t$',ylabel='Width reduction (%)',title='(b) Loss relative to nominal width')
fig.tight_layout();save(fig,'widths')
fig,ax=plt.subplots(1,2,figsize=(7,2.4),gridspec_kw={'width_ratios':[1,1.3]})
q=R['joint'][1];vals=[float(q['r1'].split()[0].strip('[')),float(q['r2'].split()[0].strip('[')),float(q['joint'].split()[0].strip('['))]
ax[0].bar(['$y=0.58$','$y=0.60$','Both'],vals,color=[colors[0],colors[2],colors[1]],width=.6)
ax[0].axhline(.02,color='black',ls='--',lw=1,label='Budget $\eta=0.02$')
ax[0].set(ylabel='Minimum $H^1$ tail norm',title='(a) Individually possible, jointly impossible')
ax[0].legend(fontsize=8)
labels=[f"({q['y1']:.2f}, {q['y2']:.2f})" for q in R['joint']]
ax[1].barh(labels,[q['ratio'] for q in R['joint']],color=colors[0],height=.6)
ax[1].axvline(1,color='black',ls='--',lw=1)
ax[1].set(xlabel='Joint budget / largest individual budget',ylabel='Requested pole heights',title='(b) Shared-data constraint across five pairs',xlim=(0,4.8))
ax[1].invert_yaxis();fig.tight_layout();save(fig,'joint')
fig,ax=plt.subplots(1,2,figsize=(7,2.5))
a=R['asymptotics'];ax[0].plot([v['tau'] for v in a],[v['ratio'] for v in a],'o-',color=colors[0],ms=4)
ax[0].axhline(1,color='black',ls='--',lw=1)
ax[0].set(xscale='log',xlabel=r'$\tau$',ylabel='Exact radius / leading asymptotic',title='(a) Parameter-free small-time prediction',ylim=(.87,1.01))
for tau,c in zip([.02,.01,.005],[colors[0],colors[1],colors[2]]):
 k=np.arange(2,110);q=np.exp(-2*tau*k*k+1.2*k)/k**2;q/=q.sum()
 ax[1].plot(k,q,color=c,label=fr'$\tau={tau:g}$')
 ax[1].axvline(.6/(2*tau),color=c,ls=':',lw=.8)
ax[1].set(xlabel='Unresolved mode $k$',ylabel='Normalized kernel contribution',title=r'(b) Hidden modes move to $k\sim y/(2\tau)$')
ax[1].legend(fontsize=8);fig.tight_layout();save(fig,'asymptotics')
print('Created four vector figures and PNG previews.')
