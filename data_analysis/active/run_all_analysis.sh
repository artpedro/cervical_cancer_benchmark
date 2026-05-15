#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BUNDLE_DIR="${REPO_ROOT}/workspace/analysis/all_metrics_dedup_bundle"
START_FROM="${1:-}"
HAS_STARTED=0

ensure_bundle() {
  if [[ -d "$BUNDLE_DIR" \
    && -f "$BUNDLE_DIR/deduplicated_best/all_summary_weighted_loss.dedup_best.csv" \
    && -f "$BUNDLE_DIR/best_checkpoints_index.csv" \
    && -d "$BUNDLE_DIR/best_checkpoints" ]]; then
    echo "[INFO] Found bundle: $BUNDLE_DIR"
    return
  fi

  echo
  echo "============================================================"
  echo "Preparing analysis bundle (missing all_metrics_dedup_bundle)"
  echo "============================================================"

  uv run data_analysis/legacy/merge_all_metrics.py --workspace-dir workspace
  uv run data_analysis/legacy/deduplicate_all_metrics_best.py --workspace-dir workspace
  uv run data_analysis/legacy/assign_ids_and_collect_best_checkpoints.py --workspace-dir workspace

  local latest_all_metrics
  latest_all_metrics="$(ls -dt workspace/all_metrics_copy_*/all_metrics 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_all_metrics}" ]]; then
    echo "[ERROR] Could not find generated all_metrics directory under workspace/all_metrics_copy_*"
    exit 1
  fi

  mkdir -p "$(dirname "$BUNDLE_DIR")"
  rm -rf "$BUNDLE_DIR"
  cp -r "$latest_all_metrics" "$BUNDLE_DIR"
  echo "[INFO] Prepared bundle at: $BUNDLE_DIR"
}

run() {
  local script_path="$1"
  if [[ -n "$START_FROM" && "$HAS_STARTED" -eq 0 ]]; then
    if [[ "$script_path" != "$START_FROM" ]]; then
      echo "[SKIP] $script_path"
      return
    fi
    HAS_STARTED=1
  fi
  echo
  echo "============================================================"
  echo "Running: $script_path"
  echo "============================================================"
  uv run "data_analysis/active/${script_path}"
}

ensure_bundle

# Core producers first (generate shared CSV artifacts).
run "evaluate_bundle_checkpoints_on_testsets.py"
run "profile_checkpoint_efficiency.py"
run "evaluate_cross_dataset_checkpoints.py"

# Downstream analysis/plots.
run "plot_cross_dataset_performance.py"
run "plot_metrics_loss_curves.py"
run "plot_aggregated_loss_curves.py"
run "plot_efficiency_vs_performance.py"
run "plot_pareto_front.py"
run "plot_macro_f1_vs_latency.py"
run "plot_f1_vs_domain_shift.py"
run "plot_test_performance_boxplots.py"
run "print_top_models_by_dataset.py"
run "plot_dataset_sample_panels.py"
run "plot_transform_probe_grid.py"

# Tables / manuscript helpers.
run "generate_test_results_tex_table.py"
run "generate_efficiency_summary_tex_table.py"
run "generate_training_hparams_tex_table.py"
run "generate_inference_efficiency_hparams_tex_table.py"
run "generate_dataset_split_summary_tex_table.py"
run "generate_binary_label_mapping_tex_table.py"
run "generate_augmentations_tex_table.py"

# Generalizability suite
run "build_generalization_master_table.py"
run "plot_transfer_heatmaps_by_regime.py"
run "plot_generalization_gap_and_robustness.py"
run "plot_efficiency_vs_generalization.py"
run "generate_generalizability_summary_table.py"

# Research-question statistical analyses (outputs under workspace/statistical_results/RQ*)
run "test_rq1_statistical_analysis.py"
run "test_rq2_statistical_analysis.py"
run "test_rq3_statistical_analysis.py"
run "test_rq4_statistical_analysis.py"
run "test_rq5_statistical_analysis.py"

if [[ -n "$START_FROM" && "$HAS_STARTED" -eq 0 ]]; then
  echo "[ERROR] START_FROM script not found in pipeline: $START_FROM"
  exit 1
fi

echo
echo "[DONE] All analysis scripts completed."
