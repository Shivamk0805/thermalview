"""Feature-set screening on the February 2021 training/validation periods only (never test).

The February test period is never scored here; January is the untouched test month."""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from thermalview import features as F, models as M
from thermalview.forecasting import make_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

src = sys.argv[1]
df = pd.read_csv(src, parse_dates=['ts']); df['node_id'] = df.node_id.astype(str)
opt = ['power_w', 'inlet_c', 'fan_rpm']
H = [5, 15, 30]
t0 = time.time()
feat = F.build_features(df, opt, H)
print('features', feat.shape, round(time.time() - t0), 's', flush=True)

# --- extra candidate features (causal, per node) ---
def extras(f):
    g = f.groupby('node_id', sort=False)
    out = pd.DataFrame(index=f.index)
    T = f['temp']; L = f['load']
    for k in (30, 60):
        out[f'temp_d{k}'] = T - g['temp'].shift(k)
        out[f'load_d{k}'] = L - g['load'].shift(k)
        out[f'load_mean{k}'] = g['load'].transform(lambda s: s.rolling(k, min_periods=k).mean())
        out[f'temp_std{k}'] = g['temp'].transform(lambda s: s.rolling(k, min_periods=k).std())
    out['temp_dev60'] = T - g['temp'].transform(lambda s: s.rolling(60, min_periods=60).mean())
    out['load_max15'] = g['load'].transform(lambda s: s.rolling(15, min_periods=15).max())
    out['load_min15'] = g['load'].transform(lambda s: s.rolling(15, min_periods=15).min())
    big = (g['load'].diff().abs() >= 10).astype(int)
    # minutes since the last load change of >= 10 points (capped at 60)
    def since(b):
        c = np.full(len(b), 60.0); last = -10**9
        for i, v in enumerate(b.to_numpy()):
            if v: last = i
            c[i] = min(60, i - last)
        return pd.Series(c, index=b.index)
    out['min_since_load_step'] = big.groupby(f['node_id'], sort=False).transform(since)
    out['hour_sin'] = np.sin(2 * np.pi * f['ts'].dt.hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * f['ts'].dt.hour / 24)
    return out

t0 = time.time()
X2 = extras(feat)
feat = pd.concat([feat, X2], axis=1).dropna(subset=list(X2.columns))
print('extras', X2.shape[1], round(time.time() - t0), 's; rows', len(feat), flush=True)

REL = F.feature_columns(opt, 'relative')
LONG = REL + ['temp_d30', 'temp_d60', 'load_d15' if 'load_d15' in feat else 'load_d30', 'load_d30', 'load_d60',
              'load_mean30', 'load_mean60', 'temp_std30', 'temp_std60']
LONG = list(dict.fromkeys(LONG))
LEVEL = LONG + ['temp_dev60', 'load_max15', 'load_min15', 'min_since_load_step']
TOD = LEVEL + ['hour_sin', 'hour_cos']
SETS = {'rel(current)': REL, 'long': LONG, 'long+shape': LEVEL, 'long+shape+tod': TOD}

def mlp():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                                                        max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=7))
def hgb_big():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=63, max_iter=600, l2_regularization=1.0,
                                         min_samples_leaf=100, early_stopping=False, random_state=7)
MODELS = {
    'ridge': lambda: M.build('ridge', {'alpha': 100.0}, 7),
    'rf': lambda: M.build('random_forest', {'max_depth': 12, 'min_samples_leaf': 20}, 7),
    'hgb': lambda: M.build('hist_gb', {'learning_rate': 0.05, 'max_leaf_nodes': 31}, 7),
    'hgb_big': hgb_big,
    'mlp': mlp,
}
res = []
for h in H:
    sp = make_split(feat, h, 0.6, 0.2, 0.2, 7)
    tr = sp.train.sample(n=min(250_000, len(sp.train)), random_state=7)
    yv = sp.val[f'y_{h}']
    pv = np.abs(yv).mean()
    for sname, cols in SETS.items():
        for mname, mk in MODELS.items():
            if mname in ('rf',) and sname not in ('rel(current)', 'long+shape'):
                continue
            t0 = time.time()
            m = mk().fit(tr[cols], tr[f'y_{h}'])
            mae = np.abs(yv - m.predict(sp.val[cols])).mean()
            res.append({'h': h, 'features': sname, 'model': mname, 'val_mae': mae, 'val_skill': 1 - mae / pv, 's': time.time() - t0})
            print(f"h={h:2d} {sname:16s} {mname:8s} val skill {1 - mae / pv:+.4f}  ({time.time() - t0:.0f}s)", flush=True)
pd.DataFrame(res).to_csv(str(__import__('pathlib').Path(__file__).with_name('feature_screening_feb2021.csv')), index=False)

# Usage:  python experiments/feature_screening.py data/processed/telemetry_feb.csv.gz
# Output: experiments/feature_screening_feb2021.csv (validation skill per horizon, feature set, model).
# Outcome on February 2021: longer history + load shape helped every tree model at 5 and 15 min;
# time of day lowered validation skill; the MLP did not beat the tree models. The pipeline's
# "extended" feature set is the "long+shape" set here.
