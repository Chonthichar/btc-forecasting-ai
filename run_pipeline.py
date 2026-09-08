
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.pipeline import (
    load_config, load_and_merge, validate_dataset, prepare_dataset,
    reserve_final_test, get_walk_forward_folds, run_baseline_cv,
    fit_final_baseline, train_torch_sequence_model,
    train_vae_transformer_fold, build_eval_context, save_torch_state,
    environment_info, set_seed,
)

def summarize_cv(df):
    metric_cols = ["accuracy","precision","recall","f1","roc_auc","pr_auc","mcc"]
    return {
        c: {
            "mean": float(df[c].mean()),
            "std": float(df[c].std(ddof=1)),
        }
        for c in metric_cols if c in df.columns
    }

def feature_sets(market_cols, sent_cols):
    return {
        "market_only": list(market_cols),
        "sentiment_only": list(sent_cols),
        "market_plus_sentiment": list(market_cols) + list(sent_cols),
    }


def vae_feature_sets(market_cols, sent_cols):
    # For sentiment_only, the VAE reconstructs the sentiment sequence itself.
    # For market_plus_sentiment, the VAE reconstructs market features and uses
    # the separate sentiment GRU branch as in the multimodal architecture.
    return {
        "market_only": (list(market_cols), []),
        "sentiment_only": (list(sent_cols), []),
        "market_plus_sentiment": (list(market_cols), list(sent_cols)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/btc_1h.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["validate","baselines","deep","all"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seeds"][0])

    run_name = cfg["experiment"].get("name") or (
        f'{cfg["asset"]}_{cfg["interval"]}_'
        f'target{cfg["experiment"].get("target_horizon_hours", 1)}h_'
        f'{cfg["experiment"]["start"].replace("-","")}_'
        f'{cfg["experiment"]["end"].replace("-","")}'
    )
    out = Path(cfg["paths"]["output_root"]) / run_name
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, out / "config.yaml")

    # 1-3 Load / validate / merge
    df = load_and_merge(cfg)
    report = validate_dataset(df, cfg)
    clean, market_cols, sent_cols = prepare_dataset(df, cfg)

    clean.to_parquet(out / f"aligned_{cfg['asset'].lower()}_market_sentiment.parquet", index=False)
    with open(out / "validation_report.json","w") as f:
        json.dump(report, f, indent=2)
    with open(out / "feature_list.json","w") as f:
        json.dump({
            "market": market_cols,
            "sentiment": sent_cols,
            "market_only_count": len(market_cols),
            "sentiment_only_count": len(sent_cols),
            "market_plus_sentiment_count": len(market_cols)+len(sent_cols),
        }, f, indent=2)
    with open(out / "environment.json","w") as f:
        json.dump(environment_info(), f, indent=2)

    # 4-6 split design
    final_gap = cfg["experiment"].get(
        "final_test_gap_hours", cfg["experiment"].get("target_horizon_hours", 0)
    )
    dev, purged_final_gap, test = reserve_final_test(
        clean, cfg["experiment"]["final_test_fraction"], final_gap
    )
    folds = get_walk_forward_folds(len(dev), cfg["cv"]["n_splits"], cfg["cv"]["gap_hours"])
    fold_manifest = []
    for i,(tr,va) in enumerate(folds,1):
        fold_manifest.append({
            "fold": i,
            "train_start": str(dev.iloc[tr]["timestamp"].min()),
            "train_end": str(dev.iloc[tr]["timestamp"].max()),
            "validation_start": str(dev.iloc[va]["timestamp"].min()),
            "validation_end": str(dev.iloc[va]["timestamp"].max()),
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
        })
    with open(out / "split_manifest.json","w") as f:
        json.dump({
            "development_start": str(dev["timestamp"].min()),
            "development_end": str(dev["timestamp"].max()),
            "final_test_start": str(test["timestamp"].min()),
            "final_test_end": str(test["timestamp"].max()),
            "walk_forward_gap_hours": cfg["cv"]["gap_hours"],
            "final_test_gap_hours": final_gap,
            "final_test_gap_start": (
                str(purged_final_gap["timestamp"].min()) if len(purged_final_gap) else None
            ),
            "final_test_gap_end": (
                str(purged_final_gap["timestamp"].max()) if len(purged_final_gap) else None
            ),
            "final_test_gap_rows": int(len(purged_final_gap)),
            "folds": fold_manifest,
        }, f, indent=2)

    print("\nDataset ready")
    print("  rows:", len(clean))
    print("  date:", clean["timestamp"].min(), "->", clean["timestamp"].max())
    print("  target:", cfg["experiment"]["target_column"])
    print("  horizon hours:", cfg["experiment"].get("target_horizon_hours"))
    print("  market features:", len(market_cols))
    print("  sentiment features:", len(sent_cols))
    print("  feature-set ablations: market_only | sentiment_only | market_plus_sentiment")
    print(
        "  deep training: max epochs", cfg["training"]["epochs"],
        "| patience", cfg["training"]["patience"],
        "| early-stop min_delta", cfg["training"].get("min_delta", 0.0),
    )
    print(
        "  dev:", len(dev),
        "| final purge gap:", len(purged_final_gap),
        "| final test:", len(test)
    )
    print("  walk-forward folds:", len(folds))

    if args.stage == "validate":
        return

    primary = cfg["evaluation"]["primary_metric"]

    # 7 Baselines + 10 ablation
    if args.stage in {"baselines", "all"}:
        all_cv_results = []
        baseline_summaries = {}
        for feature_name, features in feature_sets(market_cols, sent_cols).items():
            for model_name in cfg["models"]["baselines"]:
                seed = cfg["training"]["seeds"][0]
                cv_metrics, cv_preds, _, _ = run_baseline_cv(
                    clean, features, cfg, model_name, seed
                )
                key = f"{model_name}__{feature_name}"
                cv_metrics["model"] = model_name
                cv_metrics["feature_set"] = feature_name
                cv_metrics.to_csv(out / f"cv_{key}.csv", index=False)
                cv_preds.to_parquet(out / f"cv_predictions_{key}.parquet", index=False)
                baseline_summaries[key] = summarize_cv(cv_metrics)
                all_cv_results.append(cv_metrics)

        all_cv = pd.concat(all_cv_results, ignore_index=True)
        all_cv.to_csv(out / "all_baseline_cv_results.csv", index=False)
        with open(out / "baseline_cv_summary.json","w") as f:
            json.dump(baseline_summaries, f, indent=2)

        # choose baseline using mean primary CV metric
        ranking = (
            all_cv.groupby(["model","feature_set"])[primary]
            .mean().sort_values(ascending=False)
        )
        best_model, best_feature_set = ranking.index[0]
        best_features = feature_sets(market_cols, sent_cols)[best_feature_set]
        final_dir = out / f"final_baseline__{best_model}__{best_feature_set}"
        final_dir.mkdir(exist_ok=True)
        final_metrics = fit_final_baseline(
            dev, test, best_features, cfg, best_model,
            cfg["training"]["seeds"][0], final_dir
        )
        with open(final_dir / "metrics.json","w") as f:
            json.dump(final_metrics, f, indent=2)

        print("\nBest baseline by CV:", best_model, best_feature_set)
        print("Final holdout metrics:", final_metrics)

        if args.stage == "baselines":
            return
    # 8-9 Deep models, 5-fold CV with contiguous context across purge gaps.
    # Model selection uses CV only. The untouched final test is evaluated once
    # after choosing the best deep configuration and a CV-derived epoch count.
    deep_rows = []
    seed = cfg["training"]["seeds"][0]
    seq_len = cfg["experiment"]["sequence_length"]

    for feature_name, features in feature_sets(market_cols, sent_cols).items():
        for kind in ["lstm", "gru"]:
            for fold_id, (tr_idx, va_idx) in enumerate(folds, 1):
                tr = dev.iloc[tr_idx].copy()
                va = dev.iloc[va_idx].copy()
                val_context = build_eval_context(dev, int(va_idx[0]), seq_len)

                model, scaler, metrics, preds, info = train_torch_sequence_model(
                    tr, val_context, va, features, cfg, kind, seed
                )
                row = {
                    "model": kind,
                    "feature_set": feature_name,
                    "fold": fold_id,
                    "best_epoch": info["best_epoch"],
                    "epochs_ran": info["epochs_ran"],
                    **metrics,
                }
                deep_rows.append(row)
                preds.to_parquet(
                    out / f"cv_predictions_{kind}__{feature_name}__fold{fold_id}.parquet",
                    index=False,
                )

    for feature_name, (primary_cols, s_cols) in vae_feature_sets(market_cols, sent_cols).items():
        for fold_id, (tr_idx, va_idx) in enumerate(folds, 1):
            tr = dev.iloc[tr_idx].copy()
            va = dev.iloc[va_idx].copy()
            val_context = build_eval_context(dev, int(va_idx[0]), seq_len)

            model, ms, ss, metrics, preds, info = train_vae_transformer_fold(
                tr, val_context, va, primary_cols, s_cols, cfg, seed
            )
            row = {
                "model": "vae_transformer_v2",
                "feature_set": feature_name,
                "fold": fold_id,
                "best_epoch": info["best_epoch"],
                "epochs_ran": info["epochs_ran"],
                "validation_reconstruction_loss": info["validation_reconstruction_loss"],
                "validation_kl_loss": info["validation_kl_loss"],
                **metrics,
            }
            deep_rows.append(row)
            preds.to_parquet(
                out / f"cv_predictions_vae_transformer_v2__{feature_name}__fold{fold_id}.parquet",
                index=False,
            )

    deep_df = pd.DataFrame(deep_rows)
    deep_df.to_csv(out / "deep_cv_results.csv", index=False)

    deep_summary = (
        deep_df.groupby(["model", "feature_set"])[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
        ].agg(["mean", "std"])
    )
    deep_summary.to_csv(out / "deep_cv_summary.csv")

    # 11: select the best deep configuration using mean CV primary metric only.
    deep_ranking = (
        deep_df.groupby(["model", "feature_set"])[primary]
        .mean()
        .sort_values(ascending=False)
    )
    best_deep_model, best_deep_features = deep_ranking.index[0]
    selected_rows = deep_df[
        (deep_df["model"] == best_deep_model)
        & (deep_df["feature_set"] == best_deep_features)
    ]
    selected_epochs = max(1, int(round(selected_rows["best_epoch"].median())))

    # Build final-test context from the full clean chronology. This includes the
    # purged final-gap hour as input context, but that hour is never a train target.
    test_start_pos = int(test.index[0])
    test_context = build_eval_context(clean, test_start_pos, seq_len)

    final_deep_dir = out / f"final_deep__{best_deep_model}__{best_deep_features}"
    final_deep_dir.mkdir(exist_ok=True)

    if best_deep_model in {"lstm", "gru"}:
        final_features = feature_sets(market_cols, sent_cols)[best_deep_features]
        final_model, final_scaler, final_deep_metrics, final_preds, final_info = (
            train_torch_sequence_model(
                dev,
                test_context,
                test,
                final_features,
                cfg,
                best_deep_model,
                seed,
                epochs_override=selected_epochs,
                early_stopping=False,
            )
        )
        save_torch_state(final_model, final_deep_dir / "model_state.pt")
        import joblib
        joblib.dump(final_scaler, final_deep_dir / "scaler.joblib")
    else:
        final_primary_cols, final_sent_cols = vae_feature_sets(market_cols, sent_cols)[best_deep_features]
        (
            final_model,
            final_market_scaler,
            final_sent_scaler,
            final_deep_metrics,
            final_preds,
            final_info,
        ) = train_vae_transformer_fold(
            dev,
            test_context,
            test,
            final_primary_cols,
            final_sent_cols,
            cfg,
            seed,
            epochs_override=selected_epochs,
            early_stopping=False,
        )
        save_torch_state(final_model, final_deep_dir / "model_state.pt")
        import joblib
        joblib.dump(final_market_scaler, final_deep_dir / "primary_scaler.joblib")
        if best_deep_features != "sentiment_only":
            joblib.dump(final_market_scaler, final_deep_dir / "market_scaler.joblib")
        if final_sent_scaler is not None:
            joblib.dump(final_sent_scaler, final_deep_dir / "sentiment_scaler.joblib")

    final_preds["prediction"] = (final_preds["prob_up"] >= 0.5).astype(int)
    final_preds.to_parquet(final_deep_dir / "final_test_predictions.parquet", index=False)
    with open(final_deep_dir / "metrics.json", "w") as f:
        json.dump(final_deep_metrics, f, indent=2)
    with open(final_deep_dir / "training_info.json", "w") as f:
        json.dump({
            "selected_by": f"mean_cv_{primary}",
            "selected_model": best_deep_model,
            "selected_feature_set": best_deep_features,
            "selected_epochs_from_cv_median": selected_epochs,
            "seed": seed,
            "sequence_length": seq_len,
            "final_training_info": final_info,
        }, f, indent=2)

    print("\nDeep CV summary:")
    print(deep_summary)
    print(
        "\nBest deep configuration by CV:",
        best_deep_model,
        best_deep_features,
        "| final epochs:",
        selected_epochs,
    )
    print("Final deep holdout metrics:", final_deep_metrics)
    print("\nPipeline outputs saved to:", out)

if __name__ == "__main__":
    main()
