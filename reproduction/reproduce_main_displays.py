"""Render existing aggregate sources. No fitting, simulations or inference."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize

import argparse
_parser=argparse.ArgumentParser(description='Render manuscript displays from released aggregate sources only.')
_parser.add_argument('--output',type=Path,default=Path('reproduced_displays'))
_parser.add_argument('--figures',nargs='+',default=['1','2','3','4','5','S1'])
_args=_parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
OUT=_args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
S=ROOT/'manuscript_source_data'; F=OUT/'MAIN_FIGURES'; F.mkdir(exist_ok=True)
(OUT/'QC').mkdir(exist_ok=True); (OUT/'SUPPLEMENT').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','axes.titleweight':'bold'})
C={'M1':'#0072B2','M2':'#CC79A7','M3':'#D55E00','M4':'#333333'}
MK={'M1':'o','M2':'s','M3':'^','M4':'D'};LS={'M1':'-','M2':'--','M3':'-.','M4':':'}
DG=['G','T','S']; DL={'G':'Gaussian','T':'t5','S':'Skew-normal'}; DS=[.025,.05,.10,.20]
CON=['M3_vs_M4_repeated_realization_information','M2_vs_M3_crossed_structure','M1_vs_M2_full_covariance','M1_vs_M4_complete_strategy']
CN=['M3−M4','M2−M3','M1−M2','M1−M4']; MET=['DER','DY','CDY','FQ_all']; ML={'DER':'DER','DY':'DY','CDY':'CDY','FQ_all':'FQ-all','SER':'macro-SER','FQ_def':'macro FQ-def'}
def read(n):return pd.read_csv(S/(n+'.csv'))
def legend(): return [Line2D([0],[0],color=C[s],marker=MK[s],ls=LS[s],label=s,ms=6) for s in C]
def save(fig,n):
    if str(n) not in _args.figures:
        plt.close(fig);return
    for ext in ['pdf','svg','png']:fig.savefig((OUT/'SUPPLEMENT' if str(n)=='S1' else F)/f'Figure{n}.{ext}',dpi=600,facecolor='white')
    plt.close(fig);print('Figure',n,'exported',flush=True)

# Figure 1: declarative diagram specification is also exported.
fig=plt.figure(figsize=(14,11));ax=fig.add_axes([.035,.035,.93,.90]);ax.set(xlim=(-1,101),ylim=(0,100));ax.axis('off')
fig.suptitle('Study architecture and evaluation-strategy ladder',fontsize=19,y=.98)
nodes=[];edges=[]
def box(id,x,y,w,h,title,body,color='#555555'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=0.7',facecolor='#FAFAFA',edgecolor=color,lw=1.6))
    ax.text(x+w/2,y+h-2,title,ha='center',va='top',fontsize=11,fontweight='bold',color=color)
    ax.text(x+w/2,y+h/2-1,body,ha='center',va='center',fontsize=9.8,linespacing=1.45)
    nodes.append(dict(node_id=id,x=x,y=y,width=w,height=h,title=title,text=body))
def arrow(a,b,meaning='Process flow'):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.3,color='#555555'));edges.append({'from_xy':a,'to_xy':b,'meaning':meaning })
ax.text(0,98,'A  Clinical-data architecture',fontsize=13,fontweight='bold')
box('mimic',0,80,31,14,'MIMIC-IV v3.1','Source training\nHeld-out internal evaluation')
box('sic',34.5,80,31,14,'SICdb v1.0.8','External evaluation')
box('eicu',69,80,31,14,'eICU-CRD v2.0','Multicenter external evaluation\nEligible hospital populations')
ax.text(0,75,'B  Repeated synthesis and destination evaluation',fontsize=13,fontweight='bold')
steps=[('source','Source cohort','Training data'),('generator','Generator','Stochastic synthesis'),('realization','Realizations','Multiple datasets'),('learner','Learner','Fixed specification'),('hospital','Hospitals','Destination evaluation'),('eul','EUL matrix','Hospital × realization')]
for i,(id,t,b) in enumerate(steps):
    box(id,i*17,59,15,12,t,b)
    if i<5:arrow((i*17+15.4,65),(i*17+16.5,65))
ax.text(0,54,'C  Future-use targets and evaluation',fontsize=13,fontweight='bold')
box('targetB',0,34,47.5,16,'Primary Target B','New utility-estimable realization ×\nfuture eligible exchangeable hospital')
box('targetA',52.5,34,47.5,16,'Secondary Target A','Future eligible hospital\naveraged over realizations')
ax.text(50,30,'Known-truth scenarios: Gaussian, t5 and skew-normal heterogeneity',ha='center',fontsize=10)
ax.text(50,26.5,'Decision stringency is specified separately from model architecture',ha='center',fontsize=10)
ax.text(0,23,'D  Evaluation-strategy ladder',fontsize=13,fontweight='bold')
lad=[('M4','Single realization\nacross hospitals'),('M3','Multiple realizations\naggregated before\nhospital inference'),('M2','Crossed realization / hospital /\ninteraction representation\nDiagonal Vₕ'),('M1','Same crossed structure\nFull Vₕ')]
for i,(s,b) in enumerate(lad):box(s,i*26,2,22,15,s,b,C[s])
for i,t in enumerate(['M3−M4\nRepeated-realization\ninformation strategy contrast','M2−M3\nCrossed-structure\nstrategy contrast','M1−M2\nFull-covariance component contrast\n(cleanest component comparison)']):
    arrow((i*26+22.3,9.5),(i*26+25.7,9.5),'Strategy comparison; not a causal ablation');ax.text(i*26+24,20,t,ha='center',va='center',fontsize=7.3)
fig.text(.5,.022,'M1−M4: complete-strategy contrast. Ladder arrows denote comparisons, not causal ablations.',ha='center',fontsize=10)
(OUT/'FIGURE1_DIAGRAM_SPECIFICATION.json').write_text(json.dumps({'nodes':nodes,'arrows':edges,'strategy_contrasts':dict(zip(CN,['Repeated-realization information strategy contrast','Crossed-structure strategy contrast','Full-covariance component contrast','Complete-strategy contrast'])),'numeric_results_shown':False},indent=2))
save(fig,1)

# Figure 2: tolerance rows avoid ambiguous labels in near-coincident profiles.
d=read('FIGURE2_SOURCE_DATA');fig,axs=plt.subplots(4,3,figsize=(11,11),sharex=True,sharey=True)
fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.88,hspace=.22,wspace=.12)
fig.suptitle('Decision profiles across evaluation strategies',fontsize=17,y=.97)
fig.legend(handles=[Line2D([0],[0],color=C[s],marker=MK[s],mfc='none',ls='none',label=s,ms=7) for s in C],loc='upper center',bbox_to_anchor=(.5,.945),ncol=4,frameon=False)
for i,delta in enumerate(DS):
    for j,g in enumerate(DG):
        a=axs[i,j];q=d[(d.dgp==g)&(d.delta==delta)]
        for _,r in q.iterrows():a.scatter(r.DY,r.DER,facecolors='none',marker=MK[r.strategy_short],s={'M1':125,'M2':45,'M3':145,'M4':40}[r.strategy_short],edgecolors=C[r.strategy_short],linewidths=1.4,zorder=3)
        a.set(xlim=(-.03,1.03),ylim=(-.015,.32));a.grid(color='#E4E4E4',lw=.6,zorder=0)
        if i==0:a.set_title(DL[g],fontsize=12)
        if j==0:a.set_ylabel(f'δ = {delta:g}\nDecision-error rate',fontsize=10)
        if i==3:a.set_xlabel('Definitive yield',fontsize=11)
fig.text(.5,.025,'P95 (c = 0.95). Each marker is one absolute macro profile; tolerances are not joined.',ha='center',fontsize=10)
save(fig,2)

# Figure 3: four contrast panels, each with four aligned metric subpanels.
d=read('FIGURE3_SOURCE_DATA');fig=plt.figure(figsize=(15,15));gs=fig.add_gridspec(4,4,left=.14,right=.98,bottom=.08,top=.91,hspace=.58,wspace=.28)
fig.suptitle('Paired strategy contrasts',fontsize=20,y=.98)
fig.text(.5,.95,'Paired differences and 95% Monte Carlo intervals: comparison minus reference',ha='center',fontsize=14,fontweight='bold')
rowlabels=[f'{DL[g]}  {x:g}' for g in DG for x in DS]
lims={m:(min(0,d.loc[d.metric==m,'mc_lower'].min()),max(0,d.loc[d.metric==m,'mc_upper'].max())) for m in MET}
for k,con in enumerate(CON):
    for j,m in enumerate(MET):
        a=fig.add_subplot(gs[k,j]);q=d[(d.contrast==con)&(d.metric==m)];comp=con[:2]
        for y,(g,x) in enumerate([(g,x) for g in DG for x in DS]):
            r=q[(q.dgp==g)&(q.delta==x)].iloc[0];v=float(r.paired_difference)
            a.errorbar(v,y,xerr=[[v-r.mc_lower],[r.mc_upper-v]],fmt=MK[comp],color=C[comp],mfc='white' if r.mc_lower<=0<=r.mc_upper else C[comp],markersize=4.5,lw=1.2,capsize=2)
        a.axvline(0,color='black',lw=1,ls='--',zorder=0)
        lo,hi=lims[m];pad=(hi-lo)*.09;a.set_xlim(lo-pad,hi+pad)
        a.set_ylim(11.7,-.7);a.set_yticks(range(12),rowlabels if j==0 else ['']*12,fontsize=8.5)
        for y in [3.5,7.5]:a.axhline(y,color='#CCCCCC',lw=.7)
        a.set_title('Δ'+ML[m],fontsize=11);a.tick_params(axis='x',labelsize=9);a.grid(axis='x',color='#E5E5E5',lw=.5)
        if j==0:a.text(-.48,1.15,f'{chr(65+k)}  {CN[k]}',transform=a.transAxes,fontsize=13,fontweight='bold')
        if k==3:a.set_xlabel('Difference (proportion)',fontsize=10)
fig.text(.5,.028,'All 3 DGPs × 4 tolerances × 4 metrics in each contrast. Open markers: MC interval includes zero. Common-success N: Table 3.\nThe sign indicates change in the named metric; it does not define a universal favorable direction.',ha='center',fontsize=10,linespacing=1.5)
save(fig,3)

# Figure 4: all 576 cell colors retained, numeric cell annotations removed.
d=read('FIGURE4_SOURCE_DATA');templates=sorted(d.template_id.unique());regimes=sorted(d.regime.unique());strategies=sorted(d.strategy.unique())
fig,axs=plt.subplots(1,3,figsize=(7.2,5.9));fig.subplots_adjust(left=.245,right=.985,bottom=.205,top=.865,wspace=.11)
fig.suptitle('Coverage architecture across all template and regime cells',fontsize=10.5,y=.982)
fig.text(.5,.936,'Nominal coverage = 0.95',ha='center',fontsize=7.5)
fig.text(.5,.911,'A: future hospital averaged over realizations   B: joint future realization and hospital',ha='center',fontsize=6.8)
cols=[(s,t) for s in strategies for t in ['A','B']];rows=[(t,r) for t in templates for r in regimes]
yt=[t.replace('__',' / ').replace('LR_L2','LR')+'  '+r[:2] for t,r in rows]
coverage_rendered=[]
for j,g in enumerate(DG):
    a=axs[j];v=np.array([[d[(d.dgp==g)&(d.template_id==t)&(d.regime==r)&(d.strategy==s)&(d.target==tar)].coverage.iloc[0] for s,tar in cols] for t,r in rows])
    im=a.pcolormesh(np.arange(9),np.arange(25),v,cmap='cividis',vmin=.5,vmax=1,edgecolors='white',linewidth=.35,rasterized=False)
    a.set_ylim(24,0);a.set_xticks(np.arange(8)+.5,[f'{s[:2]}\n{t}' for s,t in cols],fontsize=6.4)
    a.set_yticks(np.arange(24)+.5,yt if j==0 else ['']*24,fontsize=6.4)
    a.set_title(DL[g],fontsize=8,pad=6);a.tick_params(length=0,pad=3)
    for y,(t,r) in enumerate(rows):
        for x,(s,tar) in enumerate(cols):
            val=v[y,x];coverage_rendered.append(dict(dgp=g,template_id=t,regime=r,strategy=s,target=tar,coverage=float(val)))
            if val<.90:a.add_patch(Rectangle((x+.075,y+.10),.85,.80,fill=False,edgecolor='white' if val<.75 else 'black',ls=(0,(2,1.4)),lw=.8))
            elif val>.99:a.add_patch(Rectangle((x+.075,y+.10),.85,.80,fill=False,edgecolor='black',lw=1.5))
    for y in range(4,24,4):a.axhline(y,color='#555555',lw=.55)
    for x in [2,4,6]:a.axvline(x,color='#555555',lw=.55)
    assert len(a.texts)==0
cax=fig.add_axes([.36,.139,.51,.022]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[.5,.6,.7,.8,.9,.95,1])
cb.ax.tick_params(labelsize=6.5,length=2,pad=2);cb.set_label('Coverage proportion',fontsize=7,labelpad=2);cb.ax.axvline(.95,color='black',lw=1.2);cb.solids.set_rasterized(False)
fig.text(.5,.060,'Dashed outline: <0.90. Solid outline: >0.99. Descriptive concentration thresholds,',ha='center',fontsize=6.7)
fig.text(.5,.041,'not inferential calibration cutoffs. Exact values: Table S8 and Figure 4 source data.',ha='center',fontsize=6.7)
fig.text(.5,.020,'R1 realization-dominant; R2 hospital-dominant; R3 interaction-dominant; R4 mixed.',ha='center',fontsize=6.4)
save(fig,4)

# Figure 5A: plot only already stored c=.99 minus c=.80 endpoint changes.
rd=read('FIGURE5A_SOURCE_DATA');dd=read('FIGURE5B_SOURCE_DATA')
fig=plt.figure(figsize=(7.2,8.7))
fig.suptitle('Decision-stringency sensitivity',fontsize=12,y=.985)
fig.text(.02,.951,'A  Change from c=0.80 to c=0.99',fontsize=9,fontweight='bold')
fig.text(.02,.931,'Sign: value at c=0.99 minus value at c=0.80',fontsize=7.5)
fig.legend(handles=[Line2D([0],[0],color=C[s],marker=MK[s],ls='none',label=s,ms=3.5) for s in C],loc='upper right',bbox_to_anchor=(.99,.953),ncol=4,frameon=False,fontsize=7,handletextpad=.3,columnspacing=.8)
ga=fig.add_gridspec(1,4,left=.185,right=.985,bottom=.550,top=.892,wspace=.25)
patterns=[(g,x) for g in DG for x in DS]
endpoint_rendered=[]
for mi,m in enumerate(['DY','CDY','DER','SER']):
    a=fig.add_subplot(ga[mi]);column=m+'_change_c099_minus_c080'
    lo=min(0,rd[column].min());hi=max(0,rd[column].max());pad=(hi-lo)*.09 or .01
    for yi,(g,x) in enumerate(patterns):
        for si,s in enumerate(C):
            r=rd[(rd.dgp==g)&(rd.delta==x)&(rd.strategy.str.startswith(s+'_'))].iloc[0]
            val=float(r[column]);a.plot(val,yi+(si-1.5)*.17,marker=MK[s],color=C[s],ms=3.0,ls='none',mew=.5,zorder=3)
            endpoint_rendered.append(dict(dgp=g,delta=x,strategy=r.strategy,metric=m,value=val))
    a.axvline(0,color='black',ls='--',lw=.7,zorder=1)
    a.set(xlim=(lo-pad,hi+pad),ylim=(11.6,-.6));a.set_title('Δ'+ML[m],fontsize=8)
    a.set_yticks(range(12),[f'{DL[g]}  {x:g}' for g,x in patterns] if mi==0 else ['']*12,fontsize=6.8)
    a.tick_params(axis='x',labelsize=6.5,length=2,pad=2);a.tick_params(axis='y',length=0,pad=3)
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3));a.grid(axis='x',color='#DDDDDD',lw=.35)
    for y in [3.5,7.5]:a.axhline(y,color='#888888',lw=.55)
    a.set_xlabel('Change (proportion)',fontsize=6.5,labelpad=3)
fig.text(.022,.498,'Rows: DGP and δ. All 48 stored endpoint changes per metric; full trajectories: Figure S1.',fontsize=6.6)

fig.text(.02,.465,'B  Strategy-contrast sensitivity to stricter decision stringency',fontsize=9,fontweight='bold')
fig.text(.02,.445,'P95: c=0.95. C95: a central-95%-interval representation of an approximately c=0.975',fontsize=6.9)
fig.text(.02,.430,'tail criterion, apart from boundary convention.',fontsize=6.9)
gb=fig.add_gridspec(1,3,left=.185,right=.985,bottom=.100,top=.396,wspace=.20)
dm=['DER','DY','CDY','FQ_all','SER','FQ_def'];symbols={'Strict reversal':'×','Same direction':'●','Null-compatible':'○'};colors={'Strict reversal':'#A34C00','Same direction':'#333333','Null-compatible':'#888888'}
classification_rendered=[]
for j,g in enumerate(DG):
    a=fig.add_subplot(gb[j]);a.set(xlim=(-.5,5.5),ylim=(15.5,-.5));a.set_title(DL[g],fontsize=8,pad=5)
    a.set_xticks(range(6),['DER','DY','CDY','FQ-all','SER','FQ-def'],fontsize=6.0)
    a.set_yticks(range(16),[f'{cn}  {x:g}' for cn in CN for x in DS] if j==0 else ['']*16,fontsize=6.8)
    for yi,(con,x) in enumerate([(con,x) for con in CON for x in DS]):
        for xi,m in enumerate(dm):
            r=dd[(dd.dgp==g)&(dd.contrast==con)&(dd.delta==x)&(dd.metric==m)].iloc[0];cl=r.display_classification
            a.text(xi,yi,symbols[cl],ha='center',va='center',color=colors[cl],fontsize=9,fontweight='bold')
            classification_rendered.append(dict(dgp=g,contrast=con,delta=x,metric=m,classification=cl))
    for y in [3.5,7.5,11.5]:a.axhline(y,color='#888888',lw=.55)
    a.set_xticks(np.arange(-.5,6,1),minor=True);a.set_yticks(np.arange(-.5,16,1),minor=True)
    a.grid(which='minor',color='#E7E7E7',lw=.35);a.tick_params(which='both',length=0,pad=3)
fig.text(.02,.065,'× Strict interval-supported reversal     ● Same interval-supported direction',fontsize=7)
fig.text(.02,.044,'○ At least one null-compatible interval. All 288 stored classifications retained.',fontsize=7)
fig.text(.02,.023,'SER and FQ-def are macro means. Related scenarios are not independent replications.',fontsize=6.5)
save(fig,5)

# Figure S1: move the complete existing confidence trajectories to the Supplement.
profiles=read('FIGURES1_SOURCE_DATA');fig=plt.figure(figsize=(7.2,11.3))
fig.suptitle('Complete confidence-grid trajectories',fontsize=12,y=.987)
fig.legend(handles=legend(),loc='upper center',bbox_to_anchor=(.5,.975),ncol=4,frameon=False,fontsize=7,markerscale=.55)
outer=fig.add_gridspec(4,3,left=.09,right=.98,bottom=.062,top=.928,hspace=.28,wspace=.20)
cs=[.8,.9,.95,.975,.99];xs=np.arange(5);trajectory_rendered=[]
for di,delta in enumerate(DS):
    for gj,g in enumerate(DG):
        inner=outer[di,gj].subgridspec(4,1,hspace=.12)
        for mi,m in enumerate(['DY','CDY','DER','SER']):
            a=fig.add_subplot(inner[mi]);q=profiles[(profiles.dgp==g)&(profiles.delta==delta)]
            for s in C:
                z=q[q.strategy_short==s].sort_values('confidence');assert len(z)==5
                a.plot(xs,z[m],ls=LS[s],marker=MK[s],color=C[s],ms=2.1,lw=.65)
                trajectory_rendered.extend(dict(dgp=g,delta=delta,strategy=s,metric=m,confidence=float(r.confidence),value=None if pd.isna(r[m]) else float(r[m])) for _,r in z.iterrows())
            a.set(ylim=(-.04,1.04),xticks=xs,yticks=[0,1]);a.tick_params(labelsize=5.8,length=1.8,pad=1.5);a.grid(color='#E0E0E0',lw=.35)
            if mi==0:a.set_title(f'{DL[g]}   δ={delta:g}',fontsize=7.5,pad=5)
            if gj==0:a.set_ylabel(ML[m],fontsize=6.5,labelpad=3)
            else:a.set_yticklabels([])
            if mi==3:a.set_xticklabels(['.80','.90','.95','.975','.99'],fontsize=6)
            else:a.set_xticklabels([])
fig.text(.5,.033,'Confidence c (equally spaced categories). All 240 existing profiles; four metrics per profile.',ha='center',fontsize=7)
fig.text(.5,.016,'Each trajectory fixes DGP, tolerance and strategy. Numerical values: Table S10.',ha='center',fontsize=7)
save(fig,'S1')

(OUT/'QC/FINAL_DISPLAY_RENDER_DATA.json').write_text(json.dumps(dict(coverage=coverage_rendered,endpoint_changes=endpoint_rendered,classifications=classification_rendered,trajectories=trajectory_rendered),indent=2)+'\n')
(OUT/'QC/FIGURE_RENDER_COUNTS.json').write_text(json.dumps({'Figure2':48,'Figure3':192,'Figure4':len(coverage_rendered),'Figure5A':len(rd),'Figure5B':len(dd),'Figure5A_plotted_metric_points':len(endpoint_rendered),'FigureS1':len(profiles),'FigureS1_plotted_metric_points':len(trajectory_rendered),'strategy_encoding':{'colors':C,'markers':MK,'line_styles':LS}},indent=2)+'\n')
