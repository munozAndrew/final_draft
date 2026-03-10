import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb
from scipy import stats

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "output_final"
ASSIGNMENTS_FILE = "output_draft/domain_assignments.csv"
MIN_VOLUME = 100
KALSHI = "data/KALSHI/markets/*.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_domain_assignments():
    if not os.path.exists(ASSIGNMENTS_FILE):
        sys.exit(f"Missing {ASSIGNMENTS_FILE}: run extract_categories.py + build_domains.py first.")
    mapping = {}
    with open(ASSIGNMENTS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["prefix"]] = row["domain"]
    print(f"Loaded {len(mapping):,} prefix assignments across {len(set(mapping.values()))} domains")
    return mapping


def load_kalshi():
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            ticker,
            event_ticker,
            title,
            result,
            CAST(volume AS BIGINT) AS volume,
            CAST(last_price AS BIGINT) AS last_price
        FROM '{KALSHI}'
        WHERE result IN ('yes', 'no')
          AND volume >= {MIN_VOLUME}
    """).df()
    con.close()
    return df

def add_prefix(df):
    df["prefix"] = df["event_ticker"].str.split("-").str[0].fillna("")
    return df


def add_word_count(df):
    clean = df["title"].fillna("").str.replace(r"\*\*", "", regex=True).str.strip()
    df["word_count"] = clean.str.split().apply(len)
    return df


def add_brier(df):
    outcome = (df["result"] == "yes").astype(float)
    prediction = df["last_price"] / 100.0
    df["brier"] = (prediction - outcome) ** 2
    return df


def filter_to_focus(df, domain_map):
    df["domain"] = df["prefix"].map(domain_map)
    df = df[df["domain"].notna()].copy()
    return df


def brier_by_bucket(df, n_bins=5):
    df = df.copy()
    df["bucket"] = pd.qcut(df["word_count"], q=n_bins, labels=False, duplicates="drop")
    result = df.groupby("bucket").agg(
        n=("brier", "size"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        wc_min=("word_count", "min"),
        wc_max=("word_count", "max"),
    ).reset_index()
    return result


def spearman_corr(df, x_col):
    rho, p = stats.spearmanr(df[x_col], df["brier"])
    return rho, p


def multivariate_ols_np(df, x_col="word_count", group_col="domain"):
    
    domain_dummies = pd.get_dummies(df[group_col], drop_first=True)
    
    ref_domain = [d for d in df[group_col].unique() if d not in domain_dummies.columns][0]

    X_df = pd.concat([df[[x_col]], domain_dummies], axis=1)
    X = np.column_stack([np.ones(len(X_df)), X_df.values.astype(np.float64)])
    y = df["brier"].values.astype(np.float64)
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y

    y_hat = X @ beta
    residuals = y - y_hat
    mse = (residuals ** 2).sum() / (n - k)

    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    coef_names = ["Intercept", x_col] + list(domain_dummies.columns)

    return dict(
        coef_names=coef_names, betas=beta, se=se,
        t_stats=t_stats, p_values=p_values,
        r2=r2, residuals=residuals, y_hat=y_hat,
        n=n, k=k, ref_domain=ref_domain,
    )


def print_ols_table(ols_result, title="Multivariate OLS Results"):
    lines = []
    lines.append(f"\n{'='*68}")
    lines.append(f"  {title}")
    lines.append(f"  n = {ols_result['n']:,}   k = {ols_result['k']}   R² = {ols_result['r2']:.4f}")
    lines.append(f"  Reference domain: {ols_result['ref_domain']}")
    lines.append(f"{'='*68}")
    lines.append(f"  {'Variable':<38} {'Beta':>8}  {'SE':>8}  {'t':>7}  {'p':>8}")
    lines.append(f"  {'-'*64}")
    
    for name, b, se, t, p in zip(
        ols_result["coef_names"], ols_result["betas"],
        ols_result["se"], ols_result["t_stats"], ols_result["p_values"]):
        
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        
        lines.append(f"  {name:<38} {b:>8.5f}  {se:>8.5f}  {t:>7.2f}  {p:>8.4f} {sig}")
        
    lines.append(f"{'='*68}")
    text = "\n".join(lines)
    print(text)

    path = f"{OUTPUT_DIR}/ols_summary.txt"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"OLS summary saved → {path}")
    return text


def fig1_brier_by_word_bucket(df):
    bucket_stats = brier_by_bucket(df)
    rho, p_rho = spearman_corr(df, "word_count")

    labels = [
        f"{int(row.wc_min)}–{int(row.wc_max)} words\n(n={int(row.n):,})"
        for _, row in bucket_stats.iterrows()
    ]
    
    se = bucket_stats["brier_std"] / np.sqrt(bucket_stats["n"])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(bucket_stats)), bucket_stats["brier_mean"],
                  color="#2563EB", alpha=0.85, width=0.6)
    
    ax.errorbar(range(len(bucket_stats)), bucket_stats["brier_mean"],
                yerr=1.96 * se, fmt="none", color="black", capsize=4, linewidth=1.5)

    for i, (bar, row) in enumerate(zip(bars, bucket_stats.itertuples())):
        ax.text(i, row.brier_mean + 0.002, f"{row.brier_mean:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(bucket_stats)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Brier Score  (lower = more accurate)", fontsize=11)

    ax.set_title(
        "Prediction Accuracy by Question Length (Word Count as Complexity Proxy)\n"
        "Kalshi: economics / politics / weather / awards markets",
        fontsize=11, fontweight="bold",
    )
    
    ax.text(0.97, 0.95, f"Spearman ρ = {rho:.3f}  (p={p_rho:.1e})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = f"{OUTPUT_DIR}/fig1_brier_by_word_bucket.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 1 saved → {path}")


def fig2_domain_comparison(df):
    domain_stats = df.groupby("domain").agg(
        n=("brier", "size"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        wc_mean=("word_count", "mean"),
    ).sort_values("brier_mean").reset_index()

    print("\nDomain stats (for writeup):")
    print(domain_stats[["domain", "n", "brier_mean", "wc_mean"]].to_string(index=False))

    colors = {
        "Economics / Finance": "#2563EB",
        "Politics / Government": "#DC2626",
        "Weather": "#16A34A",
        "Entertainment / Awards": "#D97706",
    }
    
    bar_colors = [colors.get(d, "#6B7280") for d in domain_stats["domain"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    se = domain_stats["brier_std"] / np.sqrt(domain_stats["n"])

    bars1 = ax1.barh(domain_stats["domain"], domain_stats["brier_mean"],
                     color=bar_colors, alpha=0.85)
    ax1.errorbar(domain_stats["brier_mean"], range(len(domain_stats)), xerr=1.96 * se,
                 fmt="none", color="black", capsize=4, linewidth=1.5)
    
    for bar, (_, row) in zip(bars1, domain_stats.iterrows()):
        ax1.text(row.brier_mean + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"n={int(row.n):,}", va="center", fontsize=8.5)
        
    ax1.set_xlabel("Mean Brier Score", fontsize=11)
    ax1.set_title("Accuracy by Domain", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.3)

    bars2 = ax2.barh(domain_stats["domain"], domain_stats["wc_mean"],
                     color=bar_colors, alpha=0.85)
    
    for bar, (_, row) in zip(bars2, domain_stats.iterrows()):
        ax2.text(row.wc_mean + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{row.wc_mean:.1f}w", va="center", fontsize=8.5)
    ax2.set_xlabel("Avg Word Count", fontsize=11)
    ax2.set_title("Question Length by Domain", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Domain-Level Accuracy vs. Question Length (Kalshi)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_domain_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2 saved → {path}")


def fig3_category_scatter(df, min_n=50):

    cat_stats = df.groupby("prefix").agg(
        n=("brier", "size"),
        brier_mean=("brier", "mean"),
        wc_mean=("word_count", "mean"),
        domain=("domain", "first"),
    ).query(f"n >= {min_n}").reset_index()

    colors = {
        "Economics / Finance": "#2563EB",
        "Politics / Government": "#DC2626",
        "Weather": "#16A34A",
        "Entertainment / Awards": "#D97706",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for domain, color in colors.items():
        sub = cat_stats[cat_stats["domain"] == domain]
        if sub.empty:
            continue
        ax.scatter(sub["wc_mean"], sub["brier_mean"],
                   s=np.log1p(sub["n"]) * 16,
                   color=color, alpha=0.75, edgecolors="white", linewidth=0.5,
                   label=domain)
        for _, row in sub.iterrows():
            ax.annotate(row["prefix"], (row["wc_mean"], row["brier_mean"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points", color=color)

    m, b, r, p, se_slope = stats.linregress(cat_stats["wc_mean"], cat_stats["brier_mean"])
    x_range = np.linspace(cat_stats["wc_mean"].min(), cat_stats["wc_mean"].max(), 100)

    ax.plot(x_range, m * x_range + b, color="black", linewidth=1.5, linestyle="--",
            label=f"OLS: slope={m:.4f}, r={r:.2f}, p={p:.3f}")
    print(f"\nFig 3 OLS (category level):  slope={m:.5f}  r={r:.3f}  p={p:.4f}")

    ax.set_xlabel("Mean Word Count per Category", fontsize=11)
    ax.set_ylabel("Mean Brier Score  (lower = more accurate)", fontsize=11)
    ax.set_title(
        "Category-Level Question Length vs. Accuracy (Kalshi)\n"
        "Each dot = one market category | size proportional to log(market count)",
        fontsize=11, fontweight="bold",
    )
    
    ax.legend(fontsize=9, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    path = f"{OUTPUT_DIR}/fig3_category_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 3 saved → {path}")


def fig4_ols_diagnostics(ols_result):
    residuals = ols_result["residuals"]
    y_hat = ols_result["y_hat"]
    std_resid = residuals / residuals.std()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    fig.suptitle(
        "OLS Diagnostic Plots  (brier ~ word_count + domain)\n"
        "[Issue 4: LINE assumption check]",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.scatter(y_hat, residuals, alpha=0.05, s=2, color="#2563EB")
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted\n(check: Linearity, Equal Variance)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    (qq_x, qq_y), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(qq_x, qq_y, alpha=0.1, s=3, color="#2563EB")
    ax.plot(qq_x, slope * np.array(qq_x) + intercept, color="red", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Normal QQ Plot\n(check: Normality of residuals)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    ax.scatter(y_hat, np.sqrt(np.abs(std_resid)), alpha=0.05, s=2, color="#2563EB")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.set_title("Scale-Location\n(check: Equal Variance / Homoscedasticity)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    ax.hist(residuals, bins=80, color="#2563EB", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution\n(Brier scores are right-skewed by construction)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_ols_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 4 saved → {path}")


def fig5_category_diagnostics(df, min_n=50):
    
    cat_stats = df.groupby("prefix").agg(
        n=("brier", "size"),
        brier_mean=("brier", "mean"),
        wc_mean=("word_count", "mean"),
        domain=("domain", "first"),
    ).query(f"n >= {min_n}").reset_index()

    x = cat_stats["wc_mean"].values
    y = cat_stats["brier_mean"].values
    n_cat = len(x)

    X = np.column_stack([np.ones(n_cat), x])
    k = X.shape[1]

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    mse = (residuals ** 2).sum() / (n_cat - k)

    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    cooks_d = (residuals ** 2) / (k * mse) * h / (1 - h) ** 2

    std_resid = residuals / np.sqrt(mse * (1 - h))

    colors_domain = {
        "Economics / Finance": "#2563EB",
        "Politics / Government": "#DC2626",
        "Weather": "#16A34A",
        "Entertainment / Awards": "#D97706",
    }
    pt_colors = [colors_domain.get(d, "#6B7280") for d in cat_stats["domain"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Category-Level OLS Diagnostics  (bivariate: brier_mean ~ wc_mean)\n"
        "[Issue 4: Cook's distance and standardized residuals]",
        fontsize=11, fontweight="bold",
    )

    ax1.bar(range(n_cat), cooks_d, color=pt_colors, alpha=0.8)
    cutoff = 4 / n_cat
    ax1.axhline(cutoff, color="red", linestyle="--", linewidth=1,
                label=f"Cook's cutoff = 4/n = {cutoff:.3f}")
    for i, (cd, row) in enumerate(zip(cooks_d, cat_stats.itertuples())):
        if cd > cutoff:
            ax1.text(i, cd + 0.002, row.prefix, fontsize=6, ha="center", rotation=45)
    ax1.set_xlabel("Category (sorted by index)")
    ax1.set_ylabel("Cook's Distance")
    ax1.set_title("Cook's Distance\n(influential categories above red line)")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.scatter(h, std_resid, c=pt_colors, alpha=0.8, s=40, edgecolors="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(2, color="red", linestyle="--", linewidth=1, label="|std resid| = 2")
    ax2.axhline(-2, color="red", linestyle="--", linewidth=1)
    for i, row in cat_stats.iterrows():
        if abs(std_resid[i]) > 2:
            ax2.annotate(row["prefix"], (h[i], std_resid[i]), fontsize=6)
    ax2.set_xlabel("Leverage (h_ii)")
    ax2.set_ylabel("Standardized Residual")
    ax2.set_title("Residuals vs Leverage\n(outliers outside ±2 bands)")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig5_category_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 5 saved → {path}")


if __name__ == "__main__":
    domain_map = load_domain_assignments()

    df = load_kalshi()
    df = add_prefix(df)
    df = add_word_count(df)
    df = add_brier(df)
    df = filter_to_focus(df, domain_map)

    print(f"\nTotal focus markets: {len(df):,}")

    rho, p_rho = spearman_corr(df, "word_count")
    print(f"Spearman rho (word_count vs brier): {rho:.4f}  p={p_rho:.2e}")

    ols_result = multivariate_ols_np(df)
    print_ols_table(ols_result, title="Multivariate OLS: brier ~ word_count + domain")

    fig1_brier_by_word_bucket(df)
    fig2_domain_comparison(df)
    fig3_category_scatter(df)
    fig4_ols_diagnostics(ols_result)
    fig5_category_diagnostics(df) 
