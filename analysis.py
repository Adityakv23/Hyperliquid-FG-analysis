"""
Hyperliquid × Fear & Greed Index — Full Analysis
=================================================
Run: python analysis.py
Outputs: outputs/ folder with all figures and stats tables
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings, os
warnings.filterwarnings('ignore')

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)

# ─── PALETTE ──────────────────────────────────────────────────────────────────
FEAR_C, GREED_C, NEUTRAL_C = '#E05C5C', '#5CB85C', '#F0A500'
BG, CARD, TEXT, GRID, ACCENT = '#0F1117', '#1A1D27', '#E8E8F0', '#2A2D3A', '#7B61FF'
sns.set_theme(style='dark', rc={
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID, 'grid.linewidth': 0.5
})
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

# ══════════════════════════════════════════════════════════════════════════════
# PART A — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A — Data Loading & Preparation")
print("=" * 60)

# ─── Load ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data/"   # put CSVs in data/ folder or change paths below
FG_PATH  = DATA_DIR + "fear_greed_index.csv"
HD_PATH  = DATA_DIR + "historical_data.csv"

fg = pd.read_csv(FG_PATH)
hd = pd.read_csv(HD_PATH)

print(f"\nFear & Greed Index")
print(f"  Shape       : {fg.shape[0]:,} rows × {fg.shape[1]} cols")
print(f"  Columns     : {list(fg.columns)}")
print(f"  Missing     : {fg.isnull().sum().sum()}")
print(f"  Duplicates  : {fg.duplicated().sum()}")

print(f"\nHistorical Trading Data")
print(f"  Shape       : {hd.shape[0]:,} rows × {hd.shape[1]} cols")
print(f"  Columns     : {list(hd.columns)}")
print(f"  Missing     : {hd.isnull().sum().sum()}")
print(f"  Duplicates  : {hd.duplicated().sum()}")

# ─── Clean Fear & Greed ───────────────────────────────────────────────────────
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

def simplify_sentiment(c):
    if 'Fear'  in str(c): return 'Fear'
    if 'Greed' in str(c): return 'Greed'
    return 'Neutral'

fg['sentiment'] = fg['classification'].apply(simplify_sentiment)
print(f"\nSentiment distribution (FG):\n{fg['sentiment'].value_counts().to_string()}")

# ─── Clean Historical Trades ──────────────────────────────────────────────────
# NOTE: 'Timestamp' column is unreliable (truncated precision). Use 'Timestamp IST'
#       which contains the actual trade datetime in IST format (DD-MM-YYYY HH:MM).
hd['date'] = pd.to_datetime(
    hd['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce'
).dt.normalize()

hd = hd.drop_duplicates(subset=['Account', 'Trade ID']).reset_index(drop=True)
hd['Side'] = hd['Side'].str.upper().str.strip()

print(f"\nTrading data date range: {hd['date'].min().date()} → {hd['date'].max().date()}")
print(f"Unique accounts : {hd['Account'].nunique()}")
print(f"Unique coins    : {hd['Coin'].nunique()}")

# ─── Merge ────────────────────────────────────────────────────────────────────
merged = hd.merge(fg[['date', 'value', 'classification', 'sentiment']], on='date', how='inner')
print(f"\nMerged trades   : {len(merged):,}")
print(f"Days covered    : {merged['date'].nunique()}")
print(f"Date range      : {merged['date'].min().date()} → {merged['date'].max().date()}")
print(f"\nTrades by sentiment:\n{merged['sentiment'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# PART A.3 — KEY METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A.3 — Computing Key Metrics")
print("=" * 60)

# Daily aggregates
daily = merged.groupby(['date', 'sentiment', 'value']).agg(
    total_pnl      = ('Closed PnL', 'sum'),
    trades         = ('Trade ID', 'count'),
    unique_traders = ('Account', 'nunique'),
    avg_size_usd   = ('Size USD', 'mean'),
    median_size    = ('Size USD', 'median'),
    long_trades    = ('Side', lambda x: (x == 'BUY').sum()),
    short_trades   = ('Side', lambda x: (x == 'SELL').sum()),
    win_trades     = ('Closed PnL', lambda x: (x > 0).sum()),
    total_fees     = ('Fee', 'sum'),
).reset_index()

daily['long_ratio']    = daily['long_trades']  / (daily['long_trades'] + daily['short_trades']).clip(1)
daily['win_rate']      = daily['win_trades']   / daily['trades'].clip(1)
daily['pnl_per_trade'] = daily['total_pnl']    / daily['trades'].clip(1)
daily['intensity']     = daily['trades']       / daily['unique_traders'].clip(1)

# Per-trader aggregates
trader = merged.groupby('Account').agg(
    total_pnl     = ('Closed PnL', 'sum'),
    total_trades  = ('Trade ID', 'count'),
    avg_size      = ('Size USD', 'mean'),
    win_trades    = ('Closed PnL', lambda x: (x > 0).sum()),
    pnl_std       = ('Closed PnL', 'std'),
    first_date    = ('date', 'min'),
    last_date     = ('date', 'max'),
    fear_trades   = ('sentiment', lambda x: (x == 'Fear').sum()),
    greed_trades  = ('sentiment', lambda x: (x == 'Greed').sum()),
    long_trades   = ('Side', lambda x: (x == 'BUY').sum()),
).reset_index()

trader['win_rate']      = trader['win_trades'] / trader['total_trades'].clip(1)
trader['active_days']   = (trader['last_date'] - trader['first_date']).dt.days + 1
trader['trades_per_day']= trader['total_trades'] / trader['active_days'].clip(1)
trader['fear_ratio']    = trader['fear_trades'] / trader['total_trades'].clip(1)
trader['long_ratio']    = trader['long_trades'] / trader['total_trades'].clip(1)
trader['pnl_std']       = trader['pnl_std'].fillna(0)
trader['consistency']   = trader['win_rate'] - (trader['pnl_std'] / (trader['pnl_std'].max() + 1e-9))

print("\nDaily metrics summary:")
print(daily.groupby('sentiment')[['total_pnl', 'win_rate', 'avg_size_usd', 'long_ratio', 'intensity']].mean().round(3))

# ══════════════════════════════════════════════════════════════════════════════
# PART B — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B — Analysis")
print("=" * 60)

stats = daily.groupby('sentiment').agg(
    N_days           = ('date', 'count'),
    Avg_Daily_PnL    = ('total_pnl', 'mean'),
    Median_Daily_PnL = ('total_pnl', 'median'),
    PnL_Std          = ('total_pnl', 'std'),
    Avg_Win_Rate_pct = ('win_rate', lambda x: x.mean() * 100),
    Avg_Trades_Day   = ('trades', 'mean'),
    Avg_Long_pct     = ('long_ratio', lambda x: x.mean() * 100),
    Avg_Trade_Size   = ('avg_size_usd', 'mean'),
    Avg_Intensity    = ('intensity', 'mean'),
).round(2)

print("\nCore Summary Table:")
print(stats.to_string())
stats.to_csv(f'{OUT}/summary_stats_table.csv')

# ─── Figure 1: Overview Dashboard ─────────────────────────────────────────────
print("\nGenerating figures...")
fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle('Hyperliquid × Fear & Greed — Overview Dashboard',
             fontsize=16, color=TEXT, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.4)

ax = fig.add_subplot(gs[0, 0])
counts = merged['sentiment'].value_counts().reindex(['Fear', 'Neutral', 'Greed'])
wedges, texts, autos = ax.pie(
    counts.values, labels=counts.index, autopct='%1.1f%%',
    colors=[FEAR_C, NEUTRAL_C, GREED_C], startangle=90,
    textprops={'color': TEXT, 'fontsize': 9},
    wedgeprops={'edgecolor': BG, 'linewidth': 2})
for a in autos: a.set_color(BG); a.set_fontweight('bold')
ax.set_title('Trade Distribution\nby Sentiment', color=TEXT)

ax = fig.add_subplot(gs[0, 1:3])
ax.fill_between(fg['date'], fg['value'], alpha=0.1, color=ACCENT)
ax.plot(fg['date'], fg['value'], color=ACCENT, lw=0.7, alpha=0.7)
ax.axhspan(0, 25, alpha=0.07, color=FEAR_C)
ax.axhspan(75, 100, alpha=0.07, color=GREED_C)
ax.axhline(25, color=FEAR_C, ls='--', lw=0.8, alpha=0.6, label='Fear (≤25)')
ax.axhline(75, color=GREED_C, ls='--', lw=0.8, alpha=0.6, label='Greed (≥75)')
ax.set_title('Fear & Greed Index — Full History', color=TEXT)
ax.set_ylabel('Index Value', color=TEXT); ax.set_ylim(0, 100)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=20)

for col_idx, (metric, label, ylabel) in enumerate([
    ('trades', 'Avg Daily Trades\nby Sentiment', 'Trades/Day'),
]):
    ax = fig.add_subplot(gs[0, 3])
    vals = daily.groupby('sentiment')[metric].mean().reindex(['Fear', 'Neutral', 'Greed'])
    bars = ax.bar(vals.index, vals.values, color=[FEAR_C, NEUTRAL_C, GREED_C], edgecolor=BG)
    for b, v in zip(bars, vals.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.01,
                f'{v:,.0f}', ha='center', va='bottom', color=TEXT, fontsize=8, fontweight='bold')
    ax.set_title(label, color=TEXT); ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(True, axis='y', alpha=0.3)

bottom_metrics = [
    ('pnl_per_trade', 'Avg PnL per Trade\nby Sentiment', 'USD', True),
    ('win_rate',      'Avg Daily Win Rate\nby Sentiment', '%',   False),
    ('long_ratio',    'Avg Long Ratio\nby Sentiment',    '% Long', False),
    ('avg_size_usd',  'Avg Trade Size\nby Sentiment',   'USD',  False),
]
for col_idx, (metric, title, ylabel, signed) in enumerate(bottom_metrics):
    ax = fig.add_subplot(gs[1, col_idx])
    mult = 100 if metric in ('win_rate', 'long_ratio') else 1
    vals = daily.groupby('sentiment')[metric].mean().reindex(['Fear', 'Neutral', 'Greed']) * mult
    clrs = [FEAR_C if (signed and v < 0) else c for v, c in
            zip(vals.values, [FEAR_C, NEUTRAL_C, GREED_C])]
    bars = ax.bar(vals.index, vals.values, color=clrs, edgecolor=BG)
    if signed: ax.axhline(0, color=TEXT, lw=0.6, alpha=0.5)
    if metric == 'long_ratio': ax.axhline(50, color=TEXT, ls='--', lw=0.7, alpha=0.5)
    for b, v in zip(bars, vals.values):
        fmt = f'{v:.1f}%' if metric in ('win_rate', 'long_ratio') else (f'${v:.2f}' if metric == 'pnl_per_trade' else f'${v:,.0f}')
        off = max(abs(vals))*0.02
        ax.text(b.get_x()+b.get_width()/2, v + (off if v >= 0 else -off*3),
                fmt, ha='center', va='bottom', color=TEXT, fontsize=8, fontweight='bold')
    ax.set_title(title, color=TEXT); ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(True, axis='y', alpha=0.3)

plt.savefig(f'{OUT}/fig1_overview_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("  ✓ fig1_overview_dashboard.png")

# ─── Figure 2: Performance Deep-Dive ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig.suptitle('Trader Performance: Fear vs Greed Days',
             fontsize=16, color=TEXT, fontweight='bold', y=0.98)

ax = axes[0, 0]
data_v = [merged[merged['sentiment'] == s]['Closed PnL'].clip(-500, 500).values
          for s in ['Fear', 'Neutral', 'Greed']]
parts = ax.violinplot(data_v, positions=[1,2,3], showmedians=True, showextrema=False)
for pc, c in zip(parts['bodies'], [FEAR_C, NEUTRAL_C, GREED_C]):
    pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor(BG)
parts['cmedians'].set_color(TEXT); parts['cmedians'].set_linewidth(2)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['Fear','Neutral','Greed'])
ax.axhline(0, color=TEXT, lw=0.6, alpha=0.5)
ax.set_title('PnL Distribution\n(clipped ±$500)', color=TEXT)
ax.set_ylabel('Closed PnL (USD)', color=TEXT); ax.grid(True, axis='y', alpha=0.3)

ax = axes[0, 1]
colors_map = {'Fear': FEAR_C, 'Greed': GREED_C, 'Neutral': NEUTRAL_C}
for sent in ['Fear', 'Neutral', 'Greed']:
    sub = daily[daily['sentiment'] == sent].sort_values('date')
    ax.bar(sub['date'], sub['total_pnl'], color=colors_map[sent], alpha=0.7, width=1.5, label=sent)
ax.axhline(0, color=TEXT, lw=0.6, alpha=0.5)
roll = daily.sort_values('date')['total_pnl'].rolling(14, min_periods=1).mean()
ax.plot(daily.sort_values('date')['date'], roll, color=TEXT, lw=1.2, alpha=0.7, label='14d avg')
ax.set_title('Daily Total PnL\n(colored by sentiment)', color=TEXT)
ax.set_ylabel('PnL (USD)', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=20)

ax = axes[0, 2]
wr_vals = [daily[daily['sentiment'] == s]['win_rate'].values * 100 for s in ['Fear', 'Neutral', 'Greed']]
bp = ax.boxplot(wr_vals, labels=['Fear','Neutral','Greed'], patch_artist=True,
                medianprops={'color': TEXT, 'linewidth': 2},
                flierprops={'marker':'o','markersize':3,'markerfacecolor':ACCENT,'alpha':0.4},
                whiskerprops={'color': GRID}, capprops={'color': GRID})
for p, c in zip(bp['boxes'], [FEAR_C, NEUTRAL_C, GREED_C]):
    p.set_facecolor(c); p.set_alpha(0.6); p.set_edgecolor(BG)
ax.set_title('Win Rate Distribution\nby Sentiment', color=TEXT)
ax.set_ylabel('Win Rate (%)', color=TEXT); ax.grid(True, axis='y', alpha=0.3)

ax = axes[1, 0]
dd = daily.groupby('sentiment')['total_pnl'].std().reindex(['Fear','Neutral','Greed'])
bars = ax.bar(dd.index, dd.values, color=[FEAR_C, NEUTRAL_C, GREED_C], edgecolor=BG)
for b, v in zip(bars, dd.values):
    ax.text(b.get_x()+b.get_width()/2, v+100, f'${v:,.0f}', ha='center', va='bottom',
            color=TEXT, fontsize=8, fontweight='bold')
ax.set_title('PnL Volatility\n(Drawdown Proxy — Std Dev)', color=TEXT)
ax.set_ylabel('Std Dev (USD)', color=TEXT); ax.grid(True, axis='y', alpha=0.3)

ax = axes[1, 1]
ut = daily.groupby('sentiment')['unique_traders'].agg(['mean','std']).reindex(['Fear','Neutral','Greed'])
bars = ax.bar(ut.index, ut['mean'], yerr=ut['std'], color=[FEAR_C, NEUTRAL_C, GREED_C],
              edgecolor=BG, capsize=4, error_kw={'color': TEXT, 'lw': 1.2})
for b, v in zip(bars, ut['mean']):
    ax.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}', ha='center', va='bottom',
            color=TEXT, fontsize=8, fontweight='bold')
ax.set_title('Avg Unique Traders/Day\n(±1σ)', color=TEXT)
ax.set_ylabel('Traders', color=TEXT); ax.grid(True, axis='y', alpha=0.3)

ax = axes[1, 2]
fee_pnl = daily.groupby('sentiment')[['total_pnl','total_fees']].mean().reindex(['Fear','Neutral','Greed'])
x, w = np.arange(3), 0.35
ax.bar(x-w/2, fee_pnl['total_pnl'], width=w, color=[FEAR_C,NEUTRAL_C,GREED_C], alpha=0.85, label='Avg PnL', edgecolor=BG)
ax.bar(x+w/2, -fee_pnl['total_fees'], width=w, color=[FEAR_C,NEUTRAL_C,GREED_C], alpha=0.4, label='Fee drag', edgecolor=BG, hatch='//')
ax.set_xticks(x); ax.set_xticklabels(['Fear','Neutral','Greed'])
ax.axhline(0, color=TEXT, lw=0.6, alpha=0.5)
ax.set_title('Avg PnL vs Fee Drag\nby Sentiment', color=TEXT)
ax.set_ylabel('USD', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{OUT}/fig2_performance_deepdive.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("  ✓ fig2_performance_deepdive.png")

# ─── Figure 3: Behavioral Analysis ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig.suptitle('Trader Behavior Shifts: Fear vs Greed', fontsize=16, color=TEXT, fontweight='bold', y=0.98)

ax = axes[0, 0]
daily_s = daily.sort_values('date')
for sent, c in [('Fear', FEAR_C), ('Greed', GREED_C), ('Neutral', NEUTRAL_C)]:
    sub = daily_s[daily_s['sentiment'] == sent]
    ax.scatter(sub['date'], sub['long_ratio']*100, color=c, alpha=0.3, s=10)
roll_lr = daily_s['long_ratio'].rolling(14, min_periods=1).mean()
ax.plot(daily_s['date'], roll_lr*100, color=TEXT, lw=1.2, alpha=0.8, label='14d avg')
ax.axhline(50, color=NEUTRAL_C, ls='--', lw=0.8, alpha=0.7)
fp=mpatches.Patch(color=FEAR_C, label='Fear'); gp=mpatches.Patch(color=GREED_C, label='Greed')
np_=mpatches.Patch(color=NEUTRAL_C, label='Neutral'); ll=plt.Line2D([0],[0],color=TEXT,lw=1.2,label='14d avg')
ax.legend(handles=[fp,gp,np_,ll], fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.set_title('Long Ratio Over Time\n(dots colored by sentiment)', color=TEXT)
ax.set_ylabel('% Long', color=TEXT); ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=20)

ax = axes[0, 1]
intensity = daily.groupby('sentiment')['intensity'].mean().reindex(['Fear','Neutral','Greed'])
bars = ax.bar(intensity.index, intensity.values, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG)
for b, v in zip(bars, intensity.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.1f}x', ha='center', va='bottom',
            color=TEXT, fontsize=9, fontweight='bold')
ax.set_title('Trade Intensity\n(Trades per Active Trader/Day)', color=TEXT)
ax.set_ylabel('Trades/Trader', color=TEXT); ax.grid(True, axis='y', alpha=0.3)

ax = axes[0, 2]
sc = ax.scatter(daily['value'], daily['long_ratio']*100, c=daily['value'],
                cmap='RdYlGn', alpha=0.5, s=20, vmin=0, vmax=100)
z = np.polyfit(daily['value'], daily['long_ratio']*100, 1)
xr = np.linspace(daily['value'].min(), daily['value'].max(), 100)
ax.plot(xr, np.poly1d(z)(xr), color=ACCENT, lw=2, ls='--', label=f'Trend (slope={z[0]:.3f})')
plt.colorbar(sc, ax=ax, shrink=0.8).set_label('FG Index', color=TEXT)
ax.set_title('FG Index vs Long Ratio\n(does sentiment drive longs?)', color=TEXT)
ax.set_xlabel('Fear & Greed Value', color=TEXT); ax.set_ylabel('% Long Trades', color=TEXT)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
CLIP = 15000
for sent, c, alpha in [('Fear', FEAR_C, 0.5), ('Greed', GREED_C, 0.5)]:
    d = merged[merged['sentiment'] == sent]['Size USD'].clip(0, CLIP)
    ax.hist(d, bins=60, color=c, alpha=alpha, density=True, label=sent)
    ax.axvline(d.mean(), color=c, lw=2, ls='--', label=f'{sent} μ=${d.mean():,.0f}')
ax.set_title('Trade Size Distribution\nFear vs Greed', color=TEXT)
ax.set_xlabel('Trade Size USD', color=TEXT); ax.set_ylabel('Density', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
merged['fg_bucket'] = pd.cut(merged['value'], bins=[0,25,50,75,100],
    labels=['Extreme\nFear', 'Fear /\nNeutral', 'Greed', 'Extreme\nGreed'])
merged['dow'] = pd.Categorical(merged['date'].dt.day_name(),
    categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True)
pivot = merged.groupby(['fg_bucket','dow'], observed=True)['Closed PnL'].mean().unstack()
vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
im = ax.imshow(pivot.fillna(0).values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i,j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                    color='black' if abs(val) < vmax*0.6 else 'white')
plt.colorbar(im, ax=ax, shrink=0.8).set_label('Avg PnL', color=TEXT)
ax.set_title('Avg PnL Heatmap\n(Sentiment Bucket × Day of Week)', color=TEXT)

ax = axes[1, 2]
top10 = merged.groupby('Coin')['Trade ID'].count().nlargest(10).index
cs = merged[merged['Coin'].isin(top10)].groupby(['Coin','sentiment'])['Trade ID'].count().unstack(fill_value=0)
cs_pct = cs.div(cs.sum(axis=1), axis=0).reindex(columns=['Fear','Neutral','Greed'], fill_value=0)
cs_pct = cs_pct.sort_values('Fear')
ax.barh(cs_pct.index, cs_pct['Fear'], color=FEAR_C, alpha=0.8, label='Fear')
ax.barh(cs_pct.index, cs_pct['Neutral'], left=cs_pct['Fear'], color=NEUTRAL_C, alpha=0.8, label='Neutral')
ax.barh(cs_pct.index, cs_pct['Greed'], left=cs_pct['Fear']+cs_pct['Neutral'], color=GREED_C, alpha=0.8, label='Greed')
ax.set_title('Top 10 Coins: Sentiment Mix\n(% of trades per coin)', color=TEXT)
ax.set_xlabel('Proportion', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{OUT}/fig3_behavioral_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("  ✓ fig3_behavioral_analysis.png")

# ─── Figure 4: Trader Segmentation ────────────────────────────────────────────
trader['size_seg'] = pd.qcut(trader['avg_size'],      q=3, labels=['Low Size', 'Mid Size', 'High Size'])
trader['freq_seg'] = pd.qcut(trader['trades_per_day'],q=3, labels=['Infrequent','Moderate','Frequent'])
trader['cons_seg'] = pd.qcut(trader['consistency'],   q=3, labels=['Inconsistent','Moderate','Consistent'])

w = 0.25; x3 = np.arange(3)
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig.suptitle('Trader Segmentation: Behavior × Performance × Sentiment',
             fontsize=14, color=TEXT, fontweight='bold', y=0.98)

def grouped_bar(ax, seg_col, metric_col, func, index_order, title, ylabel, signed=False):
    seg_m = merged.merge(trader[['Account', seg_col]], on='Account', how='left')
    d = seg_m.groupby([seg_col,'sentiment'])[metric_col].apply(func).unstack().reindex(index=index_order)
    for i, (s, c) in enumerate([('Fear',FEAR_C),('Neutral',NEUTRAL_C),('Greed',GREED_C)]):
        if s in d.columns:
            ax.bar(x3 + i*w, d[s], width=w, color=c, alpha=0.85, label=s, edgecolor=BG)
    ax.set_xticks(x3+w); ax.set_xticklabels(index_order, fontsize=9)
    if signed: ax.axhline(0, color=TEXT, lw=0.6, alpha=0.5)
    ax.set_title(title, color=TEXT); ax.set_ylabel(ylabel, color=TEXT)
    ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(True, axis='y', alpha=0.3)

grouped_bar(axes[0,0], 'size_seg', 'Closed PnL', 'mean',
            ['Low Size','Mid Size','High Size'], 'Avg PnL per Trade\nSize Segment × Sentiment', 'USD', True)
grouped_bar(axes[0,1], 'freq_seg', 'Closed PnL', lambda x: (x>0).mean()*100,
            ['Infrequent','Moderate','Frequent'], 'Win Rate (%)\nFrequency × Sentiment', '%')
grouped_bar(axes[0,2], 'cons_seg', 'Closed PnL', lambda x: x.sum()/1e3,
            ['Inconsistent','Moderate','Consistent'], 'Total PnL (K USD)\nConsistency × Sentiment', 'K USD', True)

ax = axes[1, 0]
seg_colors3 = [FEAR_C, NEUTRAL_C, GREED_C]
for seg, c in zip(['Low Size','Mid Size','High Size'], seg_colors3):
    sub = trader[trader['size_seg'] == seg]
    ax.scatter(sub['win_rate']*100, sub['total_pnl'].clip(-5000, 250000),
               color=c, alpha=0.5, s=40, label=seg)
ax.axhline(0, color=TEXT, lw=0.6, alpha=0.4); ax.axvline(50, color=TEXT, lw=0.6, alpha=0.4, ls='--')
ax.set_title('Win Rate vs Total PnL\n(by Size Segment)', color=TEXT)
ax.set_xlabel('Win Rate (%)', color=TEXT); ax.set_ylabel('Total PnL (USD)', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT); ax.grid(True, alpha=0.3)

grouped_bar(axes[1,1], 'freq_seg', 'Side', lambda x: (x=='BUY').mean()*100,
            ['Infrequent','Moderate','Frequent'], 'Long Bias (%)\nFrequency × Sentiment', '% Long')
axes[1,1].axhline(50, color=TEXT, ls='--', lw=0.8, alpha=0.5)

ax = axes[1, 2]
for seg, c in zip(['Inconsistent','Moderate','Consistent'], seg_colors3):
    sub = trader[trader['cons_seg'] == seg]['total_pnl'].clip(-5000, 200000)
    ax.hist(sub, bins=25, color=c, alpha=0.5, density=True,
            label=f'{seg} (μ=${sub.mean():,.0f})')
ax.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
ax.set_title('PnL Distribution\nby Consistency Segment', color=TEXT)
ax.set_xlabel('Total PnL (USD)', color=TEXT); ax.set_ylabel('Density', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{OUT}/fig4_segmentation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("  ✓ fig4_segmentation.png")

# ─── Figure 5: Clustering ──────────────────────────────────────────────────────
print("\nBONUS — Clustering")
features = ['win_rate','trades_per_day','avg_size','fear_ratio','pnl_std','long_ratio']
X = trader[features].fillna(0)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
trader['cluster'] = km.fit_predict(Xs)
cp_kmeans = trader.groupby('cluster')[features + ['total_pnl']].mean()

def label_cluster(c):
    row = cp_kmeans.loc[c]
    if row['trades_per_day'] > cp_kmeans['trades_per_day'].median() * 2:
        return 'HFT / Bot'
    elif row['avg_size'] > cp_kmeans['avg_size'].quantile(0.75):
        return 'Whale Trader'
    elif row['win_rate'] > cp_kmeans['win_rate'].median() and row['pnl_std'] < cp_kmeans['pnl_std'].median():
        return 'Disciplined Winner'
    else:
        return 'Casual Trader'

trader['archetype'] = trader['cluster'].map(label_cluster)
print("\nArchetype distribution:\n", trader['archetype'].value_counts().to_string())

cluster_colors4 = [ACCENT, FEAR_C, GREED_C, NEUTRAL_C]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle('Trader Clustering — Behavioral Archetypes (K-Means, k=4)',
             fontsize=14, color=TEXT, fontweight='bold')

ax = axes[0]
for i, c in enumerate(trader['cluster'].unique()):
    sub = trader[trader['cluster'] == c]
    ax.scatter(sub['win_rate']*100, sub['trades_per_day'].clip(0, 30),
               color=cluster_colors4[i%4], alpha=0.5, s=60, label=sub['archetype'].iloc[0],
               edgecolors=BG, linewidths=0.5)
ax.set_title('Win Rate vs Trades/Day\nby Archetype', color=TEXT)
ax.set_xlabel('Win Rate (%)', color=TEXT); ax.set_ylabel('Trades/Day (clipped 30)', color=TEXT)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT); ax.grid(True, alpha=0.3)

ax = axes[1]
apnl = trader.groupby('archetype')['total_pnl'].mean().sort_values()
ax.barh(apnl.index, apnl.values/1e3,
        color=[FEAR_C if v < 0 else GREED_C for v in apnl.values], edgecolor=BG, alpha=0.85)
ax.axvline(0, color=TEXT, lw=0.6, alpha=0.5)
ax.set_title('Avg Total PnL by Archetype\n(K USD)', color=TEXT)
ax.set_xlabel('Avg PnL (K USD)', color=TEXT); ax.grid(True, axis='x', alpha=0.3)

ax = axes[2]
radar = trader.groupby('archetype')[['win_rate','fear_ratio','long_ratio']].mean()
x_r = np.arange(3)
for i, (idx, row) in enumerate(radar.iterrows()):
    ax.bar(x_r + i*0.2, [row['win_rate']*100, row['fear_ratio']*100, row['long_ratio']*100],
           width=0.2, color=cluster_colors4[i%4], alpha=0.8, label=idx, edgecolor=BG)
ax.set_xticks(x_r+0.3); ax.set_xticklabels(['Win Rate\n(%)', '% Fear Day\nTrades', 'Long\nBias (%)'], fontsize=9)
ax.set_title('Behavioral Profile\nby Archetype', color=TEXT); ax.set_ylabel('%', color=TEXT)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT); ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.92])
plt.savefig(f'{OUT}/fig5_clustering.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("  ✓ fig5_clustering.png")

# ─── Figure 6: Predictive Model ───────────────────────────────────────────────
print("\nBONUS — Predictive Model")
dm = daily.sort_values('date').copy()
dm['next_profitable'] = (dm['total_pnl'].shift(-1) > 0).astype(int)
for lag in [1, 2, 3]:
    dm[f'pnl_lag{lag}']    = dm['total_pnl'].shift(lag)
    dm[f'fg_lag{lag}']     = dm['value'].shift(lag)
    dm[f'wr_lag{lag}']     = dm['win_rate'].shift(lag)
    dm[f'trades_lag{lag}'] = dm['trades'].shift(lag)
dm['lr_lag1'] = dm['long_ratio'].shift(1)
dm = dm.dropna(subset=['next_profitable'] + [f'pnl_lag{l}' for l in [1,2,3]])

feat_cols = ['value','trades','win_rate','long_ratio','avg_size_usd',
             'pnl_lag1','pnl_lag2','pnl_lag3','fg_lag1','fg_lag2','fg_lag3',
             'wr_lag1','trades_lag1','lr_lag1']
feat_cols = [c for c in feat_cols if c in dm.columns]
Xm = dm[feat_cols].fillna(0); ym = dm['next_profitable']

rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=4, min_samples_leaf=3)
cv_k = min(5, max(2, len(ym)//5))
if len(ym) >= 10:
    cv_scores = cross_val_score(rf, Xm, ym, cv=cv_k, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
else:
    cv_scores = np.array([0.5])
rf.fit(Xm, ym)
fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle(
    f'Predictive Model: Next-Day Profitability\nRandom Forest — CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}',
    fontsize=13, color=TEXT, fontweight='bold')

ax = axes[0]
ax.barh(fi.index, fi.values,
        color=[FEAR_C if v < fi.median() else GREED_C for v in fi.values], edgecolor=BG, alpha=0.85)
ax.set_title('Feature Importances', color=TEXT); ax.set_xlabel('Importance', color=TEXT)
ax.grid(True, axis='x', alpha=0.3)

ax = axes[1]
sc = ax.scatter(dm['value'], dm['pnl_lag1'].clip(-50000, 50000),
                c=dm['next_profitable'], cmap='RdYlGn', alpha=0.6, s=30, vmin=0, vmax=1)
ax.axvline(50, color=NEUTRAL_C, ls='--', lw=1, alpha=0.6, label='FG=50')
ax.axhline(0, color=TEXT, lw=0.6, alpha=0.5)
ax.set_title('FG Index vs Lag-1 PnL\n(green = next day profitable)', color=TEXT)
ax.set_xlabel('Fear & Greed Value', color=TEXT); ax.set_ylabel('Prev Day PnL (USD)', color=TEXT)
plt.colorbar(sc, ax=ax, shrink=0.8, ticks=[0,1]).set_label('Next Profitable', color=TEXT)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.88])
plt.savefig(f'{OUT}/fig6_predictive_model.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("  ✓ fig6_predictive_model.png")

print("\n" + "=" * 60)
print(f"All outputs saved to ./{OUT}/")
print("=" * 60)
