"""
Compare regenerated SPARC hyperparameter caches against fRAR Table 2 values.
"""
from argparse import ArgumentParser
from os.path import isfile, join

import joblib

TABLE2 = {
    "NN_gobs_gbar.p": {
        "width": 16,
        "dropout_rate": 0.05,
    },
    "ET_gobs_gbar.p": {
        "estimator__n_estimators": 92,
        "estimator__max_depth": 8,
        "estimator__min_samples_split": 2,
        "estimator__max_features": "sqrt",
        "estimator__min_impurity_decrease": 4.2e-05,
        "estimator__ccp_alpha": 4.4e-12,
        "estimator__max_samples": 0.94,
    },
    "XGB_gobs_gbar.p": {
        "estimator__n_estimators": 125,
        "estimator__max_depth": 4,
        "estimator__booster": "dart",
        "estimator__learning_rate": 0.94,
        "estimator__gamma": 1.6,
        "estimator__min_child_weight": 2.14,
        "estimator__subsample": 0.56,
    },
}

parser = ArgumentParser()
parser.add_argument("--hyper-dir", default="../results/hyper_sparc_optuna")
args = parser.parse_args()


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


for filename, expected in TABLE2.items():
    path = join(args.hyper_dir, filename)
    print(f"\n{filename}")
    if not isfile(path):
        print(f"  missing: {path}")
        continue

    got = joblib.load(path).get("best_params", {})
    for key, exp in expected.items():
        obs = got.get(key, "<missing>")
        status = "OK" if obs == exp else "DIFF"
        if isinstance(obs, float) and isinstance(exp, float):
            status = "OK" if abs(obs - exp) <= max(1e-12, 1e-6 * abs(exp)) else "DIFF"
        print(f"  {key}: got {fmt(obs)}; table2 {fmt(exp)} [{status}]")
