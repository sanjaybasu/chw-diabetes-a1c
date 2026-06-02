"""
Figure generation. Reads output/results.json (from analysis.py) and the cohort, writes
to output/figures/:
  figure1_treatment_effects.{png,pdf}  primary/concordant estimates + baseline>=9% trajectory
  efigure2_ps_distributions.{png,pdf}   propensity-score overlap
  efigure3_cost_effectiveness.{png,pdf} cost-effectiveness plane (Monte Carlo)
  efigure1_participant_flow.{png,pdf}   participant flow
  efigure4_sensitivity_forest.{png,pdf} sensitivity-analysis forest
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"; RAW = DATA/"raw"; OUT = ROOT/"output"; FIG = OUT/"figures"; FIG.mkdir(parents=True, exist_ok=True)
res = json.load(open(OUT/"results.json"))
COV = ['age','baseline_a1c','risk_percentile','comorbidity_count','pre_ed','pre_ip','pre_pcp','has_bh','has_htn','has_chf','has_pulm','polypharmacy','high_ed_ip']

pub = pd.read_parquet(DATA/"analytic_cohort.parquet"); pub['index_date']=pd.to_datetime(pub['index_date'])
labs = pd.read_parquet(RAW/"cohort372_a1c_labs.parquet"); labs['collection_date']=pd.to_datetime(labs['collection_date'])
L = labs.merge(pub[['way_id','index_date']].rename(columns={'way_id':'person_id'}),on='person_id',how='inner')
b=L[(L.collection_date>=L.index_date-pd.DateOffset(months=6))&(L.collection_date<L.index_date)].groupby('person_id')['a1c'].mean().rename('bl_mean')
f=L[(L.collection_date>=L.index_date+pd.Timedelta(days=90))&(L.collection_date<=L.index_date+pd.Timedelta(days=365))].groupby('person_id')['a1c'].mean().rename('fu_mean')
d=pub.merge(b,left_on='way_id',right_index=True,how='left').merge(f,left_on='way_id',right_index=True,how='left'); d['ych']=d['fu_mean']-d['bl_mean']
cov=[c if c!='baseline_a1c' else 'bl_mean' for c in COV]
dd=d.dropna(subset=cov+['ych']).copy()
X=dd[cov].values.astype(float); T=dd['treated'].values.astype(float)
ps=LogisticRegression(max_iter=1000).fit(X,T).predict_proba(X)[:,1]; dd=dd[(ps>=.05)&(ps<=.95)].copy()

# ===== Figure 1 =====
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[1.2,1]})
P=res['primary']; Lst=res['primary_concordance_last']
est=[("DML, mean HbA1c\n(primary, N=369)",P['ate'],P['ci_lo'],P['ci_hi']),
     ("DML, last HbA1c\n(concordance, N=369)",Lst['ate'],Lst['ci_lo'],Lst['ci_hi'])]
ates=[e[1] for e in est]; lo=[e[2] for e in est]; hi=[e[3] for e in est]; yp=np.arange(len(est))
ax1.barh(yp,ates,xerr=[[a-l for a,l in zip(ates,lo)],[h-a for a,h in zip(ates,hi)]],height=0.5,
         color=["#2171b5","#6baed6"],edgecolor="black",linewidth=0.5,capsize=4,error_kw={"linewidth":1.2})
ax1.axvline(0,color="black",linewidth=0.8); ax1.set_yticks(yp); ax1.set_yticklabels([e[0] for e in est],fontsize=9); ax1.invert_yaxis()
ax1.set_xlabel("Estimated Effect on HbA1c (percentage points)",fontsize=10); ax1.set_title("A",fontsize=14,fontweight="bold",loc="left")
for i,(a,l,h) in enumerate(zip(ates,lo,hi)): ax1.text(0.12,i,f"{a:.2f} [{l:.2f}, {h:.2f}]",va="center",ha="left",fontsize=8.5)
ax1.set_xlim(-1.0,0.9)
sub=dd[dd['bl_mean']>=9.0]; tq=sub[sub.treated==1]; cq=sub[sub.treated==0]
ax2.errorbar([0,1],[tq.bl_mean.mean(),tq.fu_mean.mean()],yerr=[tq.bl_mean.sem(),tq.fu_mean.sem()],fmt="-o",color="#d62728",linewidth=2.5,markersize=10,capsize=5,label=f"Treated (N={len(tq)})")
ax2.errorbar([0,1],[cq.bl_mean.mean(),cq.fu_mean.mean()],yerr=[cq.bl_mean.sem(),cq.fu_mean.sem()],fmt="-s",color="#2171b5",linewidth=2.5,markersize=10,capsize=5,label=f"Control (N={len(cq)})")
ax2.set_xticks([0,1]); ax2.set_xticklabels(["Baseline","Follow-up"],fontsize=10); ax2.set_ylabel("Mean HbA1c (%)",fontsize=10)
ax2.set_title("B",fontsize=14,fontweight="bold",loc="left"); ax2.legend(loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=2,fontsize=9,frameon=False)
plt.tight_layout(rect=[0,0.05,1,1])
for ext in ["png","pdf"]: fig.savefig(FIG/f"figure1_treatment_effects.{ext}",dpi=300,bbox_inches="tight")
plt.close()

# ===== eFigure 2: PS distributions =====
fig,ax=plt.subplots(figsize=(8,5))
bins=np.linspace(0,1,21)
ax.hist(ps[T==1],bins=bins,alpha=0.6,color="#d62728",label=f"Treated (N={int(T.sum())})",density=True)
ax.hist(ps[T==0],bins=bins,alpha=0.6,color="#2171b5",label=f"Control (N={int((1-T).sum())})",density=True)
ax.set_xlabel("Estimated propensity score",fontsize=11); ax.set_ylabel("Density",fontsize=11)
ax.set_title("Propensity Score Distributions",fontsize=12,fontweight="bold"); ax.legend(fontsize=10)
plt.tight_layout()
for ext in ["png","pdf"]: fig.savefig(FIG/f"efigure2_ps_distributions.{ext}",dpi=300,bbox_inches="tight")
plt.close()

# ===== eFigure 3: CE plane =====
np.random.seed(42); NS=10000
fig,ax=plt.subplots(figsize=(8,6))
recode=[("mi",0.21350,0.082,45000,0.055),("stroke",0.33650,0.055,55000,0.164),("renal",0.13690,0.045,85000,0.078),
        ("vision",0.14490,0.075,15000,0.050),("chf",0.20920,0.065,32000,0.075),("death",0.16590,0.065,0,1.0)]
for color,marker,label,eff in [("#2171b5","o",f"Mean HbA1c (primary, ATE = {P['ate']:.2f} pp)",P),
                                ("#6baed6","s",f"Last HbA1c (sensitivity, ATE = {Lst['ate']:.2f} pp)",Lst)]:
    se=(eff['ci_hi']-eff['ci_lo'])/(2*1.96); qd=[]; cd=[]
    for _ in range(NS):
        a1c=np.random.normal(eff['ate'],se); ts=0.0; tq=0.0
        for _c,beta,br,cp,qd_ in recode:
            arr=br*(1-np.exp(beta*a1c)); ts+=arr*cp; tq+=arr*qd_*10
        months=np.random.triangular(84,118,163)/30.5
        net=np.random.triangular(31.81,67.74,147.33)*months-np.random.triangular(49.69,252.70,519.67)*months-ts
        cd.append(net); qd.append(tq)
    ax.scatter(qd[::10],cd[::10],alpha=0.15,s=8,color=color,marker=marker)
    ax.scatter(np.mean(qd),np.mean(cd),s=120,color=color,marker=marker,edgecolors="black",linewidths=1,label=label,zorder=5)
ax.axhline(0,color="gray",linewidth=0.5,linestyle="--"); ax.axvline(0,color="gray",linewidth=0.5,linestyle="--")
qr=np.linspace(0,max(0.15,ax.get_xlim()[1]),100); ax.plot(qr,qr*50000,color="green",linewidth=1,linestyle=":",alpha=0.5,label="WTP = $50,000/QALY")
ax.set_xlabel("Incremental QALYs",fontsize=11); ax.set_ylabel("Incremental Net Cost ($)",fontsize=11)
ax.set_title("Cost-Effectiveness Plane",fontsize=12,fontweight="bold"); ax.legend(fontsize=9,loc="upper left")
plt.tight_layout()
for ext in ["png","pdf"]: fig.savefig(FIG/f"efigure3_cost_effectiveness.{ext}",dpi=300,bbox_inches="tight")
plt.close()

# ===== eFigure 1: participant flow =====
fig,ax=plt.subplots(figsize=(9.5,9)); ax.axis("off"); ax.set_xlim(0,10.6); ax.set_ylim(0,15)
box=lambda x,y,w,h,t,fc="#eef3f8":(ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",fc=fc,ec="black",linewidth=1)),ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.2))
arrow=lambda x1,y1,x2,y2:ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color="black",linewidth=1))
side=lambda x,y,w,h,t:(ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc="#f7f7f7",ec="gray",linewidth=0.8)),ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.6))
box(2.5,12.8,5,1.9,"Adults (>=18y) with diabetes\n(ICD-10 E08-E13 or HbA1c >=6.5%)\nidentified for care-team outreach\nVirginia Medicaid MCO, 2022-2025"); arrow(5,12.8,5,12.2)
box(2.5,10.6,5,1.6,"Identified-for-outreach diabetes cohort\nwith >=1 baseline HbA1c\n(eligible for analysis)")
side(8.2,10.7,2.1,1.4,"Treated = care team\nactivated\nControl = targeted,\nnot activated"); arrow(5,10.6,5,9.8)
side(8.2,8.7,2.1,1.6,"Excluded: no follow-up\nHbA1c 90-365 d after\nindex (addressed by\nIPCW sensitivity)"); arrow(5,9.8,5,9.0); arrow(5,9.4,8.1,9.4)
box(2.5,7.4,5,1.6,"With >=1 baseline and >=1 follow-up HbA1c\n(>=90 d after index): N = 372"); arrow(5,7.4,5,6.6)
side(8.2,6.5,2.1,1.3,"Excluded: missing\ncovariate (N=1);\nPS outside [0.05,0.95]\n(N=2)"); arrow(5,6.6,5,5.8); arrow(5,6.2,8.1,6.2)
box(2.5,4.2,5,1.6,"Propensity-score-trimmed cohort\nN = 369 (116 treated, 253 control)",fc="#dde9f5"); arrow(5,4.2,5,3.4)
box(2.0,1.6,6,1.8,"Primary analytic cohort\nMean HbA1c, follow-up 90-365 d\nN = 369 (116 treated, 253 control)",fc="#cfe0f1")
ax.text(5,0.7,"Participant flow.",ha="center",fontsize=8.5,style="italic")
for ext in ["png","pdf"]: fig.savefig(FIG/f"efigure1_participant_flow.{ext}",dpi=300,bbox_inches="tight")
plt.close()

# ===== eFigure 4: sensitivity forest =====
g=res['window_grid']
rows=[("Primary: mean HbA1c, 90-365 d (N=369)",P['ate'],P['ci_lo'],P['ci_hi']),
      ("Last value, 90-365 d (N=369)",Lst['ate'],Lst['ci_lo'],Lst['ci_hi']),
      ("Mean, 0-6 mo (N=261)",g['mean_followup_0_6mo']['mean']['ate'],g['mean_followup_0_6mo']['mean']['ci_lo'],g['mean_followup_0_6mo']['mean']['ci_hi']),
      ("Mean, 90 d-6 mo (N=210)",g['mean_followup_90d_6mo']['mean']['ate'],g['mean_followup_90d_6mo']['mean']['ci_lo'],g['mean_followup_90d_6mo']['mean']['ci_hi']),
      ("IPCW-weighted (N=369)",res['ipcw']['ipcw_weighted_primary']['ate'],res['ipcw']['ipcw_weighted_primary']['ci_lo'],res['ipcw']['ipcw_weighted_primary']['ci_hi']),
      ("Time-aligned index (N=%d)"%res['time_aligned']['n'],res['time_aligned']['ate'],res['time_aligned']['ci_lo'],res['time_aligned']['ci_hi']),
      ("Grace-period target trial (N=%d)"%res['target_trial']['n'],res['target_trial']['ate'],res['target_trial']['ci_lo'],res['target_trial']['ci_hi']),
      ("Baseline HbA1c >=8%",res['subgroups']['baseline_ge8']['ate'],res['subgroups']['baseline_ge8']['ci_lo'],res['subgroups']['baseline_ge8']['ci_hi'])]
fig,ax=plt.subplots(figsize=(9,5.5)); yp=np.arange(len(rows))[::-1]
for y,(lab,a,l,h) in zip(yp,rows):
    ax.plot([l,h],[y,y],color="#2171b5",linewidth=1.6); ax.plot(a,y,"o",color="#08519c",markersize=7)
    ax.text(1.05,y,f"{a:.2f} [{l:.2f}, {h:.2f}]",va="center",fontsize=8)
ax.axvline(0,color="black",linewidth=0.8); ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in rows],fontsize=8.5)
ax.set_xlabel("Estimated Effect on HbA1c (percentage points)",fontsize=10); ax.set_xlim(-2.4,2.0)
ax.set_title("Sensitivity analyses",fontsize=10,fontweight="bold")
plt.tight_layout()
for ext in ["png","pdf"]: fig.savefig(FIG/f"efigure4_sensitivity_forest.{ext}",dpi=300,bbox_inches="tight")
plt.close()
print("Figures written to", FIG)
