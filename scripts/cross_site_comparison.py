#!/usr/bin/env python
"""Generate combined cross-site outputs from results/<site>/outputs/*.csv"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
COMBINED    = RESULTS_DIR / "combined"
FIG_DIR     = COMBINED / "figures"

CANONICAL = ["medan","pekanbaru","pontianak","bontang","samarinda",
             "balikpapan","makassar","surabaya","kupang","jayapura"]

SITE_META = {
    "medan":{"province":"North Sumatra","regime":"equatorial_wet","lat":3.595,"lon":98.672},
    "pekanbaru":{"province":"Riau","regime":"equatorial_wet","lat":0.507,"lon":101.448},
    "pontianak":{"province":"West Kalimantan","regime":"equatorial_maritime","lat":-0.026,"lon":109.342},
    "bontang":{"province":"East Kalimantan","regime":"equatorial_maritime","lat":0.133,"lon":117.500},
    "samarinda":{"province":"East Kalimantan","regime":"equatorial_maritime","lat":-0.502,"lon":117.153},
    "balikpapan":{"province":"East Kalimantan","regime":"equatorial_maritime","lat":-1.265,"lon":116.831},
    "makassar":{"province":"South Sulawesi","regime":"monsoonal","lat":-5.147,"lon":119.432},
    "surabaya":{"province":"East Java","regime":"tropical_monsoon","lat":-7.258,"lon":112.752},
    "kupang":{"province":"East Nusa Tenggara","regime":"semi_arid_tropical","lat":-10.177,"lon":123.607},
    "jayapura":{"province":"Papua","regime":"equatorial_rainforest","lat":-2.534,"lon":140.718},
}
REGIME_COLORS = {
    "equatorial_wet":"#1B7837","equatorial_maritime":"#2166AC",
    "monsoonal":"#D6604D","tropical_monsoon":"#E08214",
    "semi_arid_tropical":"#8B1A1A","equatorial_rainforest":"#41AB5D",
}

def rc(s): return REGIME_COLORS.get(SITE_META.get(s,{}).get("regime",""),"#888")

def safe(p):
    try: return pd.read_csv(p)
    except: return None

def discover_sites():
    avail=[s for s in CANONICAL if (RESULTS_DIR/s/"outputs").exists()
           and any((RESULTS_DIR/s/"outputs").glob("*.csv"))]
    avail+=[p.name for p in RESULTS_DIR.iterdir() if p.is_dir()
            and p.name not in CANONICAL and p.name!="combined"
            and (p/"outputs").exists()]
    return avail

def save_fig(fig,name):
    path=FIG_DIR/name; fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white")
    print(f"  ✓ {name} ({os.path.getsize(path)//1024}KB)"); plt.close(fig)

def regime_handles(sites):
    seen=set(); h=[]
    for s in sites:
        r=SITE_META.get(s,{}).get("regime","")
        if r and r not in seen:
            h.append(mpatches.Patch(color=REGIME_COLORS.get(r,"#888"),alpha=0.85,
                label=r.replace("_"," ").title())); seen.add(r)
    return h

def main():
    COMBINED.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"serif","font.size":9,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":0.2})

    sites = discover_sites()
    print(f"Processing {len(sites)} sites: {sites}")

    # ── Collect data ────────────────────────────────────────────────────
    rows_lk,rows_ols,rows_wf,rows_wk,rows_shap,rows_desc=[],[],[],[],[],[]
    for s in sites:
        base = RESULTS_DIR/s/"outputs"
        try:
            df=pd.read_parquet(RESULTS_DIR/s/"data"/"01_nasa_power_clean.parquet")
            rows_desc.append({"site":s,"GHI_mean":round(df["GHI"].mean(),3),
                "GHI_sd":round(df["GHI"].std(),3),"CLOUD_mean":round(df["CLOUD"].mean(),2),
                "T2M_mean":round(df["T2M"].mean(),3),"PRECTOT_mean":round(df["PRECTOT"].mean(),3),
                "n_obs":len(df)})
        except: pass
        try:
            lk=safe(base/"02_leakage_demonstration.csv")
            if lk is not None:
                det=float(lk[lk["target_type"]=="deterministic"]["R2"].values[0])
                sto=float(lk[lk["target_type"]=="stochastic"]["R2"].values[0])
                rows_lk.append({"site":s,"R2_det":round(det,6),
                    "R2_stoch":round(sto,4),"leakage_ratio":round(det/sto,4)})
        except: pass
        try:
            ols=safe(base/"05_ols_coefficients.csv")
            if ols is not None:
                g=ols[ols["Feature"]=="GHI_anom"].iloc[0]
                rows_ols.append({"site":s,"beta":round(float(g["Coef"]),5),
                    "se":round(float(g["HC3_SE"]),5),"t_stat":round(float(g["t_stat"]),3),
                    "p":round(float(g["p_value"]),6)})
        except: pass
        try:
            mc=safe(base/"09_model_comparison_table.csv")
            if mc is not None:
                for model in ["OLS-HC3","XGBoost","SARIMAX+ONI"]:
                    row=mc[mc["Model"]==model]
                    if not row.empty:
                        r=row.iloc[0]
                        rows_wf.append({"site":s,"model":model,
                            "RMSE":round(float(r["RMSE"]),5),
                            "SkillScore":round(float(r["SkillScore"]),4),
                            "R2":round(float(r["R2"]),4)})
        except: pass
        try:
            wk=safe(base/"09_wilcoxon_test_results.csv")
            if wk is not None:
                for _,row in wk.iterrows():
                    rows_wk.append({"site":s,"model":str(row.iloc[0]).strip(),
                        "wilcoxon_p":float(row["p"]),
                        "n_pos_folds":int(row["n_positive_folds"])})
        except: pass
        try:
            sh=safe(base/"08_shap_feature_summary.csv")
            if sh is not None:
                vc=sh.columns[1] if len(sh.columns)>1 else sh.columns[0]
                top=sh.sort_values(vc,ascending=False).iloc[0]
                rows_shap.append({"site":s,"rank1":str(top.iloc[0]),
                    "shap_val":round(float(top[vc]),5)})
        except: pass

    # ── Save CSVs ─────────────────────────────────────────────────────
    df_desc=pd.DataFrame(rows_desc)
    df_lk=pd.DataFrame(rows_lk).sort_values("leakage_ratio")
    df_ols=pd.DataFrame(rows_ols)
    df_wf=pd.DataFrame(rows_wf)
    df_wk=pd.DataFrame(rows_wk)
    df_shap=pd.DataFrame(rows_shap)

    if not df_desc.empty: df_desc.to_csv(COMBINED/"dataset_summary.csv",index=False)
    if not df_lk.empty:   df_lk.to_csv(COMBINED/"cross_site_leakage.csv",index=False)
    if not df_ols.empty:  df_ols.to_csv(COMBINED/"ols_coefficients_all_sites.csv",index=False)
    if not df_wf.empty:   df_wf.to_csv(COMBINED/"walkforward_comparison.csv",index=False)
    if not df_wk.empty:   df_wk.to_csv(COMBINED/"wilcoxon_all_sites.csv",index=False)
    if not df_shap.empty: df_shap.to_csv(COMBINED/"shap_rank1.csv",index=False)

    # ── Meta-analysis ─────────────────────────────────────────────────
    if not df_ols.empty and len(df_ols)>=3:
        b=df_ols["beta"].values; se=df_ols["se"].values
        w=1/se**2; W=w.sum(); mb=(w*b).sum()/W; mse=(1/W)**0.5
        mz=mb/mse
        Q=float((w*(b-mb)**2).sum()); k=len(b)
        Qp=float(1-stats.chi2.cdf(Q,df=k-1))
        I2=max(0,(Q-(k-1))/Q*100)
        ghi_sds=[df_desc.set_index("site").loc[s,"GHI_sd"]
                 for s in df_ols["site"] if s in df_desc["site"].values]
        lk_rats=[df_lk.set_index("site").loc[s,"leakage_ratio"]
                 for s in df_ols["site"] if s in df_lk["site"].values]
        rho=float(stats.spearmanr(ghi_sds,lk_rats)[0]) if len(ghi_sds)>=3 else np.nan
        np.random.seed(42)
        if len(ghi_sds)>=3:
            arr=np.array(lk_rats)
            p_perm=sum(abs(stats.spearmanr(ghi_sds,np.random.permutation(arr))[0])>=abs(rho)
                       for _ in range(50000))/50000
        else: p_perm=np.nan
        xgb_wk=df_wk[df_wk["model"]=="XGBoost"]
        ss_groups=[df_wf[(df_wf["site"]==s)&(df_wf["model"]=="XGBoost")]["SkillScore"].values
                   for s in sites if not df_wf[(df_wf["site"]==s)&(df_wf["model"]=="XGBoost")].empty]
        kw_h,kw_p=stats.kruskal(*ss_groups) if len(ss_groups)>=3 else (np.nan,np.nan)

        pd.DataFrame([{"n_sites":len(sites),
            "meta_beta":round(mb,5),"meta_se":round(mse,5),"meta_z":round(mz,3),
            "meta_CI_lo":round(mb-1.96*mse,5),"meta_CI_hi":round(mb+1.96*mse,5),
            "Q":round(Q,4),"Q_p":round(Qp,4),"I2":round(I2,1),
            "KW_H_xgb_between_sites":round(kw_h,3),"KW_p_xgb_between_sites":round(kw_p,4),
            "spearman_rho_ghisd_leakage":round(rho,4),
            "spearman_p_permutation":round(p_perm,4),
            "leakage_min":round(df_lk["leakage_ratio"].min(),3),
            "leakage_max":round(df_lk["leakage_ratio"].max(),3),
        }]).to_csv(COMBINED/"statistical_summary.csv",index=False)
        print(f"  Meta β={mb:.5f} z={mz:.2f}  I²={I2:.1f}%")
        print(f"  Leakage {df_lk['leakage_ratio'].min():.2f}×–{df_lk['leakage_ratio'].max():.2f}×")
        print(f"  Spearman ρ={rho:.3f}, perm-p={p_perm:.4f}")
        print(f"  KW between-sites H={kw_h:.3f}, p={kw_p:.4f}")

    # ── Figures ────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    valid=[s for s in CANONICAL if s in sites]

    if not df_lk.empty:
        fig,ax=plt.subplots(figsize=(11,5))
        colors=[rc(s) for s in df_lk["site"]]
        ax.bar(range(len(df_lk)),df_lk["leakage_ratio"],color=colors,alpha=0.85,edgecolor="white")
        ax.set_xticks(range(len(df_lk)))
        ax.set_xticklabels([s.capitalize() for s in df_lk["site"]],rotation=25,ha="right")
        for i,v in enumerate(df_lk["leakage_ratio"]):
            ax.text(i,v+0.04,f"{v:.2f}×",ha="center",fontsize=8.5,fontweight="bold")
        ax.set_ylabel("Leakage Ratio (R²_det / R²_stoch)")
        ax.set_title(f"Deterministic Leakage Ratio — {len(df_lk)} Sites\n(Sorted ascending; R²_det≈1.0 at all sites)")
        ax.legend(handles=regime_handles(df_lk["site"]),fontsize=8,ncol=2)
        ax.set_facecolor("#F7F9FC"); plt.tight_layout()
        save_fig(fig,"fig02_leakage_ratio.png")

    if not df_ols.empty and len(df_ols)>=3:
        fig,ax=plt.subplots(figsize=(9,max(5,len(df_ols)*0.6+1)))
        y=np.arange(len(df_ols)); colors2=[rc(s) for s in df_ols["site"]]
        for i,(_,row) in enumerate(df_ols.iterrows()):
            ax.errorbar(row["beta"],i,xerr=1.96*row["se"],fmt="none",ecolor="gray",capsize=4,lw=1.5)
            ax.scatter(row["beta"],i,s=110,color=colors2[i],zorder=5,edgecolor="white",lw=1)
            lab="p<0.0001" if row["p"]<0.0001 else f"p={row['p']:.4f}"
            ax.text(row["beta"],i+0.22,lab,ha="center",fontsize=7.5)
        ax.axvline(mb,color="darkred",lw=2,ls="-",label=f"Meta β={mb:.4f}")
        ax.axvspan(mb-1.96*mse,mb+1.96*mse,alpha=0.15,color="darkred",label="95% CI")
        ax.axvline(0,color="black",lw=0.8,ls="--")
        ax.set_yticks(y); ax.set_yticklabels([s.capitalize() for s in df_ols["site"]])
        ax.set_xlabel("GHI_anom β (95% CI, OLS-HC3 low-VIF)")
        ax.set_title(f"GHI Anomaly Forest Plot — {len(df_ols)} Sites\n"
                     f"Meta β={mb:.4f} (I²={I2:.1f}%, p_Q={Qp:.3f})")
        ax.legend(fontsize=8.5); ax.set_ylim(-0.7,len(df_ols)-0.3)
        ax.set_facecolor("#F7F9FC"); plt.tight_layout()
        save_fig(fig,"fig03_ghi_anom_forest_plot.png")

    if not df_wf.empty:
        wf_pivot=df_wf.pivot_table(index="site",columns="model",values="SkillScore",aggfunc="first")
        col_order=[c for c in ["OLS-HC3","XGBoost","SARIMAX+ONI"] if c in wf_pivot.columns]
        wf_pivot=wf_pivot[col_order]
        ordered_rows=[s for s in CANONICAL if s in wf_pivot.index]+[s for s in wf_pivot.index if s not in CANONICAL]
        wf_pivot=wf_pivot.reindex(ordered_rows)
        fig,ax=plt.subplots(figsize=(9,max(5,len(wf_pivot)*0.55+2)))
        im=ax.imshow(wf_pivot.values,cmap="RdYlGn",vmin=-0.15,vmax=0.35,aspect="auto")
        ax.set_xticks(range(len(col_order))); ax.set_xticklabels(col_order)
        ax.set_yticks(range(len(wf_pivot))); ax.set_yticklabels([s.capitalize() for s in wf_pivot.index])
        for i in range(wf_pivot.shape[0]):
            for j in range(wf_pivot.shape[1]):
                v=wf_pivot.values[i,j]
                if not np.isnan(v):
                    ax.text(j,i,f"{v:+.3f}",ha="center",va="center",fontsize=9,
                            color="white" if abs(v)>0.15 else "black")
        plt.colorbar(im,ax=ax,shrink=0.8,label="Skill Score")
        ax.set_title(f"Walk-Forward Skill Score — {len(wf_pivot)} Sites × {len(col_order)} Models")
        plt.tight_layout(); save_fig(fig,"fig04_skillscore_heatmap.png")

    if not df_wk.empty:
        xgb_wk=df_wk[df_wk["model"]=="XGBoost"].copy()
        if len(xgb_wk)>0:
            xgb_s=xgb_wk.sort_values("wilcoxon_p").reset_index(drop=True)
            fig,ax=plt.subplots(figsize=(10,max(5,len(xgb_s)*0.5+2)))
            colors_w=[rc(s) for s in xgb_s["site"]]
            ax.barh(range(len(xgb_s)),-np.log10(xgb_s["wilcoxon_p"]+1e-10),
                    color=colors_w,alpha=0.85,edgecolor="white")
            ax.axvline(-np.log10(0.05),color="red",lw=2,ls="--",label="α=0.05")
            ax.set_yticks(range(len(xgb_s)))
            ax.set_yticklabels([s.capitalize() for s in xgb_s["site"]])
            ax.set_xlabel("−log₁₀(Wilcoxon p-value)")
            ax.set_title(f"XGBoost vs Climatology Significance — {len(xgb_s)} Sites")
            for i,(_,row) in enumerate(xgb_s.iterrows()):
                sig="**" if row["wilcoxon_p"]<0.01 else("*" if row["wilcoxon_p"]<0.05 else "ns")
                ax.text(-np.log10(row["wilcoxon_p"]+1e-10)+0.08,i,
                        f"p={row['wilcoxon_p']:.3f} ({int(row['n_pos_folds'])}/9) {sig}",
                        va="center",fontsize=8)
            ax.legend(handles=regime_handles(list(xgb_s["site"]))+[
                plt.Line2D([0],[0],color="red",lw=2,ls="--",label="α=0.05")],fontsize=8)
            ax.set_facecolor("#F7F9FC"); plt.tight_layout()
            save_fig(fig,"fig11_wilcoxon_significance.png")

    if not df_desc.empty and not df_lk.empty:
        desc_idx=df_desc.set_index("site"); lk_idx=df_lk.set_index("site")
        common=[s for s in df_ols["site"] if s in desc_idx.index and s in lk_idx.index]
        if len(common)>=3:
            xs=np.array([desc_idx.loc[s,"GHI_sd"] for s in common])
            ys=np.array([lk_idx.loc[s,"leakage_ratio"] for s in common])
            fig,ax=plt.subplots(figsize=(9,6))
            for i,s in enumerate(common):
                c=rc(s); ax.scatter(xs[i],ys[i],s=140,color=c,edgecolor="white",lw=1.5,zorder=5)
                ax.annotate(s.capitalize(),(xs[i],ys[i]),xytext=(7,4),textcoords="offset points",
                            fontsize=8.5,color=c,fontweight="bold")
            m,b_=np.polyfit(xs,ys,1); xfit=np.linspace(xs.min(),xs.max(),60)
            ax.plot(xfit,m*xfit+b_,"--",color="#555",lw=1.5,alpha=0.7)
            ax.set_xlabel("GHI Standard Deviation (kWh/m²/day)")
            ax.set_ylabel("Leakage Ratio")
            ax.set_title(f"Spatial Leakage Robustness (n={len(common)})\n"
                         f"Spearman ρ={rho:.3f}, permutation p={p_perm:.4f}")
            ax.legend(handles=regime_handles(common),fontsize=8)
            ax.set_facecolor("#F7F9FC"); plt.tight_layout()
            save_fig(fig,"fig08_leakage_vs_ghi_variance.png")

    print(f"\n✅ Cross-site comparison complete → results/combined/")

if __name__=="__main__":
    main()
