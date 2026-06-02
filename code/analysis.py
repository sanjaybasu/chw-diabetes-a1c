"""
Primary analysis — CHW Diabetes A1c.

Single, final analysis pipeline. The primary outcome is the change in *mean* HbA1c
from baseline (6 months before the index date) to follow-up (mean of values 90-365
days after the index date), estimated by doubly-robust debiased machine learning
(LinearDML) on the propensity-score-trimmed cohort (N=369).

Produces (output/results.json):
  - Covariate balance (unmatched + 1:1 PS-matched SMDs)        [eTable 1]
  - Propensity-score overlap                                    [eFigure 1 inputs]
  - Primary (mean HbA1c) + concordant single-value estimate; follow-up-window grid  [Table 2; eTable 7]
  - Heterogeneity: treatment x baseline interaction, GRF feature importance,
    group average treatment effects by CATE quartile, best linear projection  [eTables 2, 4]
  - Baseline-HbA1c subgroups; E-value; negative-control placebo
  - HbA1c measurement-count distribution by group and window   [eTable 9]
  - Inverse-probability-of-censoring weighting for follow-up availability  [eTable 8]
  - Time-zero alignment (common identification-date index) + immortal-time interval

Inputs are produced by extract_cohort.py into data/raw/ (PHI; git-ignored), plus the
base cohort data/analytic_cohort.parquet. Deterministic (random seed 42).
"""
from __future__ import annotations
import json, warnings, tempfile, os
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; RAW = DATA / "raw"; OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)

COV = ['age','baseline_a1c','risk_percentile','comorbidity_count','pre_ed','pre_ip','pre_pcp',
       'has_bh','has_htn','has_chf','has_pulm','polypharmacy','high_ed_ip']
SEED = 42

def _atomic_json(obj, path):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

def lgbm():
    return LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1)

def dml_fit(Y, T, W, sample_weight=None):
    est = LinearDML(model_y=lgbm(), model_t=LogisticRegression(max_iter=1000),
                    discrete_treatment=True, cv=5, random_state=SEED)
    est.fit(Y, T, X=None, W=W, sample_weight=sample_weight)
    ate = float(est.ate()); ci = est.ate_interval(alpha=0.05)
    sd = float(np.sqrt((Y[T==1].var() + Y[T==0].var())/2))
    se = (float(ci[1]) - float(ci[0]))/(2*1.96)
    p = float(2*(1-stats.norm.cdf(abs(ate/se)))) if se>0 else np.nan
    return dict(n=int(len(Y)), n_treated=int(T.sum()), n_control=int((1-T).sum()),
                ate=round(ate,4), ci_lo=round(float(ci[0]),4), ci_hi=round(float(ci[1]),4),
                cohens_d=round(abs(ate)/sd,3), p_value=round(p,4), pooled_sd=round(sd,4))

def smd(t, c):
    pooled = np.sqrt((np.nanvar(t) + np.nanvar(c)) / 2)
    return round(abs(np.nanmean(t) - np.nanmean(c)) / pooled, 3) if pooled > 0 else 0.0

# ============================================================
# Load cohort + A1c labs
# ============================================================
pub = pd.read_parquet(DATA/"analytic_cohort.parquet")
pub['index_date'] = pd.to_datetime(pub['index_date'])
labs = pd.read_parquet(RAW/"cohort372_a1c_labs.parquet")
labs['collection_date'] = pd.to_datetime(labs['collection_date'])
L = labs.merge(pub[['way_id','index_date']].rename(columns={'way_id':'person_id'}), on='person_id', how='inner').sort_values('collection_date')

def window_values(lo, hi):
    b = L[(L.collection_date>=L.index_date-pd.DateOffset(months=6)) & (L.collection_date<L.index_date)]
    f = L[(L.collection_date>=L.index_date+pd.Timedelta(days=lo)) & (L.collection_date<=L.index_date+pd.Timedelta(days=hi))]
    bm = b.groupby('person_id')['a1c'].agg(bl_mean='mean', bl_last='last', bl_cnt='count')
    fm = f.groupby('person_id')['a1c'].agg(fu_mean='mean', fu_last='last', fu_cnt='count')
    return pub.merge(bm, left_on='way_id', right_index=True, how='left').merge(fm, left_on='way_id', right_index=True, how='left')

def prep(d, ycol, basecol):
    cov = [c if c!='baseline_a1c' else basecol for c in COV]
    df = d.dropna(subset=cov+[ycol]).copy()
    X = df[cov].values.astype(float); T = df['treated'].values.astype(float)
    ps = LogisticRegression(max_iter=1000).fit(X, T).predict_proba(X)[:,1]
    df = df[(ps>=0.05)&(ps<=0.95)].copy()
    return df, cov

results = {}

# ============================================================
# 0. COVARIATE BALANCE (unmatched + 1:1 PS-matched) and PS OVERLAP  [eTable 1; eFigure 1]
# ============================================================
full = pub.dropna(subset=COV).copy()
Xf = full[COV].values.astype(float); Tf = full['treated'].values.astype(float)
ps_full = LogisticRegression(max_iter=1000).fit(Xf, Tf).predict_proba(Xf)[:,1]
full = full.assign(ps=ps_full)
pt, pc = ps_full[Tf==1], ps_full[Tf==0]
results['propensity_score'] = {
    'ps_treated': [round(float(pt.min()),3), round(float(pt.max()),3), round(float(pt.mean()),3)],
    'ps_control': [round(float(pc.min()),3), round(float(pc.max()),3), round(float(pc.mean()),3)],
    'common_support': [round(float(max(pt.min(), pc.min())),3), round(float(min(pt.max(), pc.max())),3)],
    'n_trimmed': int(((ps_full<0.05)|(ps_full>0.95)).sum())}
# 1:1 nearest-neighbour matching on logit(ps), caliper 0.05 SD of logit(ps).
# Independent balance diagnostic; residual matched imbalance motivates the use of DML
# over matching alone. (Unmatched SMDs are the substantive balance reported in Table 1.)
full['logit_ps'] = np.log(full['ps']/(1-full['ps']))
tr_df, co_df = full[Tf==1].copy(), full[Tf==0].copy()
nn = NearestNeighbors(n_neighbors=1).fit(co_df[['logit_ps']]); dist, idx = nn.kneighbors(tr_df[['logit_ps']])
cal = 0.05*full['logit_ps'].std()
keep = dist[:,0] <= cal
m_t = tr_df[keep]; m_c = co_df.iloc[idx[keep,0]]
bal = {c: {'unmatched': smd(full[Tf==1][c], full[Tf==0][c]), 'matched': smd(m_t[c], m_c[c])} for c in COV}
results['covariate_balance'] = bal

# ============================================================
# 1. PRIMARY (mean, [90,365]) + concordant last + follow-up-window grid
# ============================================================
grid = {}
for lo,hi,nm in [(0,183,'mean_followup_0_6mo'),(90,183,'mean_followup_90d_6mo'),(90,365,'mean_followup_90d_12mo')]:
    d = window_values(lo,hi); d['ych']=d['fu_mean']-d['bl_mean']; d['ychL']=d['fu_last']-d['bl_last']
    dm,cov = prep(d,'ych','bl_mean'); rm = dml_fit(dm['ych'].values, dm['treated'].values.astype(float), dm[cov].values.astype(float))
    dl,covl = prep(d,'ychL','bl_last'); rl = dml_fit(dl['ychL'].values, dl['treated'].values.astype(float), dl[covl].values.astype(float))
    grid[nm] = {'mean': rm, 'last': rl}
results['window_grid'] = grid
results['primary'] = grid['mean_followup_90d_12mo']['mean']
results['primary_concordance_last'] = grid['mean_followup_90d_12mo']['last']
print(f"PRIMARY (mean, 90-365d): ATE {results['primary']['ate']} "
      f"({results['primary']['ci_lo']},{results['primary']['ci_hi']}) p={results['primary']['p_value']} N={results['primary']['n']}")

dP = window_values(90,365); dP['ych']=dP['fu_mean']-dP['bl_mean']
dPt, covP = prep(dP, 'ych', 'bl_mean')
Yp = dPt['ych'].values.astype(float); Tp = dPt['treated'].values.astype(float); Wp = dPt[covP].values.astype(float)

# ============================================================
# 2. HETEROGENEITY (interaction, GRF, GATEs, BLP) + subgroups
# ============================================================
Xr = dPt[covP].copy(); Xr['T']=Tp; Xr['TxB']=Tp*dPt['bl_mean'].values
ols = sm.OLS(Yp, sm.add_constant(Xr.astype(float))).fit(cov_type='HC1')
inter = {'coef': round(float(ols.params['TxB']),3), 'p': round(float(ols.pvalues['TxB']),4),
         'ci_lo': round(float(ols.conf_int().loc['TxB'][0]),3), 'ci_hi': round(float(ols.conf_int().loc['TxB'][1]),3)}
cf = CausalForestDML(model_y=lgbm(), model_t=LogisticRegression(max_iter=1000), discrete_treatment=True,
                     cv=5, n_estimators=1000, min_samples_leaf=10, random_state=SEED)
cf.fit(Yp, Tp, X=Wp, W=Wp)
feat_imp = {c: round(float(v),3) for c,v in zip(covP, cf.feature_importances_)}
cates = cf.effect(Wp); cate_q = pd.qcut(cates, 4, labels=['Q1','Q2','Q3','Q4'])
gates = {}
for q in ['Q1','Q2','Q3','Q4']:
    m = np.asarray(cate_q==q); ci = cf.ate_interval(Wp[m], alpha=0.05); sub = dPt[m]
    gates[q] = dict(n=int(m.sum()), n_treated=int(sub['treated'].sum()), gate=round(float(cf.ate(Wp[m])),3),
                    ci_lo=round(float(ci[0]),3), ci_hi=round(float(ci[1]),3),
                    baseline_a1c=round(float(sub['bl_mean'].mean()),1), mean_age=round(float(sub['age'].mean()),1))
yhat = cross_val_predict(lgbm(), Wp, Yp, cv=5)
that = cross_val_predict(LogisticRegression(max_iter=1000), Wp, Tp, cv=5, method='predict_proba')[:,1]
blp = sm.OLS(Yp-yhat, sm.add_constant(np.column_stack([Tp-that, (Tp-that)*(cates-cates.mean())]))).fit(cov_type='HC1')
results['heterogeneity'] = {'interaction': inter, 'grf_feature_importance': dict(sorted(feat_imp.items(), key=lambda x:-x[1])),
                            'gates': gates, 'blp_coef': round(float(blp.params[2]),3), 'blp_p': round(float(blp.pvalues[2]),4)}
subg = {}
for thr in [8.0, 9.0]:
    s = dPt[dPt['bl_mean']>=thr]
    if s['treated'].sum()>=5 and (1-s['treated']).sum()>=5:
        subg[f'baseline_ge{int(thr)}'] = dml_fit(s['ych'].values, s['treated'].values.astype(float), s[covP].values.astype(float)) | {'mean_baseline': round(float(s['bl_mean'].mean()),2)}
results['subgroups'] = subg
print(f"interaction {inter['coef']} (p={inter['p']}); GRF top {max(feat_imp,key=feat_imp.get)}={max(feat_imp.values())}; BLP p={results['heterogeneity']['blp_p']}")

# ============================================================
# 3. E-VALUE + NEGATIVE CONTROL (pre-period PCP placebo)
# ============================================================
ate = results['primary']['ate']; sd = results['primary']['pooled_sd']
d_e = abs(ate)/sd; rr = np.exp(0.91*d_e); ev = rr+np.sqrt(rr*(rr-1))
ci_null = min(abs(results['primary']['ci_lo']), abs(results['primary']['ci_hi']))
rrc = np.exp(0.91*ci_null/sd); evc = rrc+np.sqrt(rrc*(rrc-1)) if rrc>1 else 1.0
results['e_value'] = {'point': round(float(ev),2), 'ci_bound': round(float(evc),2), 'rr_point': round(float(rr),2), 'cohens_d': round(float(d_e),3)}
pc_cov = [c for c in covP if c!='pre_pcp']
nc = dml_fit(dPt['pre_pcp'].values.astype(float), Tp, dPt[pc_cov].values.astype(float))
results['negative_control'] = nc | {'outcome':'pre_period_pcp_visits', 'null_included': bool(nc['ci_lo']<=0<=nc['ci_hi'])}
print(f"E-value {results['e_value']['point']}; negative control ATE {nc['ate']} (null included {results['negative_control']['null_included']})")

# ============================================================
# 4. HbA1c MEASUREMENT-COUNT DISTRIBUTION by group/window
# ============================================================
dC = window_values(90,365)
def desc(s):
    s = s.fillna(0)
    return dict(mean=round(float(s.mean()),2), median=float(s.median()), q1=float(s.quantile(.25)),
                q3=float(s.quantile(.75)), min=int(s.min()), max=int(s.max()))
mc = {}
for grp,lab_ in [(1,'treated'),(0,'control')]:
    g = dC[dC['treated']==grp]
    mc[lab_] = {'n':int(len(g)), 'baseline_count':desc(g['bl_cnt']), 'followup_count':desc(g['fu_cnt']),
                'pct_baseline_ge2':round(float((g['bl_cnt']>=2).mean()*100),1),
                'pct_followup_ge2':round(float((g['fu_cnt']>=2).mean()*100),1)}
mc['baseline_count_group_diff_p'] = round(float(mannwhitneyu(dC[dC.treated==1]['bl_cnt'].fillna(0), dC[dC.treated==0]['bl_cnt'].fillna(0)).pvalue),3)
mc['followup_count_group_diff_p'] = round(float(mannwhitneyu(dC[dC.treated==1]['fu_cnt'].fillna(0), dC[dC.treated==0]['fu_cnt'].fillna(0)).pvalue),3)
results['measurement_counts'] = mc

# ============================================================
# 5. IPCW for follow-up availability (selection bias)
# ============================================================
atrisk = pd.read_parquet(RAW/"atrisk_cohort.parquet")
atrisk['has_pulm'] = ((atrisk['copd']==1)|(atrisk['asthma']==1)).astype(int)
atrisk = atrisk.rename(columns={'any_bh':'has_bh','htn':'has_htn','chf':'has_chf'})
CCOV = ['age','baseline_a1c','risk_percentile','has_bh','has_htn','has_chf','has_pulm','polypharmacy','high_ed_ip']
ar = atrisk.dropna(subset=CCOV).copy()
inc = ar.groupby('treated')['observed'].agg(['sum','count','mean'])
_,pinc,_,_ = chi2_contingency(pd.crosstab(ar.treated, ar.observed))
DESIGN = CCOV+['treated']
Xc = sm.add_constant(ar[DESIGN].astype(float))
cens = sm.Logit(ar['observed'].astype(int), Xc).fit(disp=0)
inc_by_t = {1: float(inc.loc[1,'mean']), 0: float(inc.loc[0,'mean'])}
Xp369 = sm.add_constant(dPt[DESIGN].astype(float), has_constant='add')[Xc.columns]
p_obs = cens.predict(Xp369).clip(0.02, 0.98)
w369 = dPt['treated'].map(inc_by_t).values / p_obs.values
results['ipcw'] = {
    'at_risk_n': int(len(ar)), 'observed_n': int(ar['observed'].sum()),
    'inclusion_rate_treated': round(inc_by_t[1],3), 'inclusion_rate_control': round(inc_by_t[0],3),
    'inclusion_diff_p': round(float(pinc),4),
    'predictors_of_followup': {k: {'coef':round(float(cens.params[k]),3),'p':round(float(cens.pvalues[k]),4),'OR':round(float(np.exp(cens.params[k])),2)} for k in ['treated','baseline_a1c','age']},
    'primary_unweighted': results['primary'],
    'ipcw_weighted_primary': dml_fit(Yp, Tp, Wp, sample_weight=w369),
    'weight_summary': {'min':round(float(w369.min()),2),'median':round(float(np.median(w369)),2),'max':round(float(w369.max()),2)},
    'note': 'Censoring model fit on a contemporaneously reconstructed eligible cohort (targeted Virginia adults with diabetes and a baseline HbA1c, N=%d). Stabilized inverse-probability-of-censoring weights applied to the primary analytic cohort.' % len(ar)}
print(f"IPCW: inclusion treated {inc_by_t[1]:.3f} vs control {inc_by_t[0]:.3f} (p={results['ipcw']['inclusion_diff_p']}); "
      f"weighted ATE {results['ipcw']['ipcw_weighted_primary']['ate']}")

# ============================================================
# 6. TIME-ZERO ALIGNMENT + immortal time + grace-period target-trial emulation
# ============================================================
# Both groups anchored at the identification (first-targeting) date. To retain power,
# baseline is the most-proximate HbA1c within 12 months before to 90 days after
# identification (most engaged members' first HbA1c is recorded near identification),
# and follow-up is the mean over 90-365 days after identification.
tsd = pd.read_parquet(RAW/"treated_status_dates.parquet")
for c in ['targeted_at','activated_at','index_date']: tsd[c]=pd.to_datetime(tsd[c])
imm = (tsd['activated_at']-tsd['targeted_at']).dt.days.dropna()
results['immortal_time'] = dict(median=float(imm.median()), q1=float(imm.quantile(.25)), q3=float(imm.quantile(.75)),
                                mean=round(float(imm.mean()),1), min=int(imm.min()), max=int(imm.max()), n=int(len(imm)))
tmap = tsd.set_index('person_id')['targeted_at'].to_dict()
amap = tsd.set_index('person_id')['activated_at'].to_dict()
pub2 = pub.copy()
pub2['id_date'] = pd.to_datetime(pub2.apply(lambda r: tmap.get(r['way_id'], r['index_date']) if r['treated']==1 else r['index_date'], axis=1))
pub2['act_gap'] = pub2['way_id'].map(lambda w: (amap.get(w)-tmap.get(w)).days if (w in amap and w in tmap and pd.notna(amap.get(w)) and pd.notna(tmap.get(w))) else np.nan)
La = labs.merge(pub2[['way_id','id_date']].rename(columns={'way_id':'person_id'}), on='person_id', how='inner')
bla = La[(La.collection_date>=La.id_date-pd.Timedelta(days=365))&(La.collection_date<La.id_date+pd.Timedelta(days=90))].groupby('person_id')['a1c'].last().rename('bl_mean')
fua = La[(La.collection_date>=La.id_date+pd.Timedelta(days=90))&(La.collection_date<=La.id_date+pd.Timedelta(days=365))].groupby('person_id')['a1c'].mean().rename('fu_mean')
da = pub2.merge(bla,left_on='way_id',right_index=True,how='left').merge(fua,left_on='way_id',right_index=True,how='left')
da['ych']=da['fu_mean']-da['bl_mean']
# (a) time-zero-aligned (as-treated): treatment = ever activated
dat, cova = prep(da,'ych','bl_mean')
results['time_aligned'] = dml_fit(dat['ych'].values, dat['treated'].values.astype(float), dat[cova].values.astype(float))
# (b) grace-period target-trial emulation: treatment = activation within 365 d of identification
#     (avoids classifying on the future; later activators enter the comparison arm -> conservative)
da['T_tt'] = ((da['treated']==1) & (da['act_gap']<=365)).astype(int)
GP = 365
dtt = da.dropna(subset=cova+['ych']).copy()
ps_tt = LogisticRegression(max_iter=1000).fit(dtt[cova].values.astype(float), dtt['T_tt'].values.astype(float)).predict_proba(dtt[cova].values.astype(float))[:,1]
dtt = dtt[(ps_tt>=0.05)&(ps_tt<=0.95)].copy()
results['target_trial'] = dml_fit(dtt['ych'].values, dtt['T_tt'].values.astype(float), dtt[cova].values.astype(float)) | {'grace_period_days': GP}
print(f"immortal {results['immortal_time']['median']}d; time-aligned ATE {results['time_aligned']['ate']} N={results['time_aligned']['n']}; "
      f"target-trial ATE {results['target_trial']['ate']} (p={results['target_trial']['p_value']}) N={results['target_trial']['n']}")

_atomic_json(results, OUT/"results.json")
print(f"\nSaved {OUT/'results.json'}")
