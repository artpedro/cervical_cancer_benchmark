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

resolve_script_path() {
  local script_path="$1"
  case "$script_path" in
    evaluate/*|plots/*|tables/*|generalizability/*|statistics/*)
      printf '%s' "$script_path"
      ;;
    evaluate_bundle_checkpoints_on_testsets.py) printf '%s' "evaluate/evaluate_bundle_checkpoints_on_testsets.py" ;;
    profile_checkpoint_efficiency.py) printf '%s' "evaluate/profile_checkpoint_efficiency.py" ;;
    evaluate_cross_dataset_checkpoints.py) printf '%s' "evaluate/evaluate_cross_dataset_checkpoints.py" ;;
    build_generalization_master_table.py) printf '%s' "generalizability/build_generalization_master_table.py" ;;
    plot_transfer_heatmaps_by_regime.py) printf '%s' "generalizability/plot_transfer_heatmaps_by_regime.py" ;;
    plot_generalization_gap_and_robustness.py) printf '%s' "generalizability/plot_generalization_gap_and_robustness.py" ;;
    plot_efficiency_vs_generalization.py) printf '%s' "generalizability/plot_efficiency_vs_generalization.py" ;;
    generate_generalizability_summary_table.py) printf '%s' "generalizability/generate_generalizability_summary_table.py" ;;
    test_rq1_statistical_analysis.py) printf '%s' "statistics/test_rq1_statistical_analysis.py" ;;
    test_rq2_statistical_analysis.py) printf '%s' "statistics/test_rq2_statistical_analysis.py" ;;
    test_rq3_statistical_analysis.py) printf '%s' "statistics/test_rq3_statistical_analysis.py" ;;
    test_rq4_statistical_analysis.py) printf '%s' "statistics/test_rq4_statistical_analysis.py" ;;
    test_rq5_statistical_analysis.py) printf '%s' "statistics/test_rq5_statistical_analysis.py" ;;
    generate_*)
      printf 'tables/%s' "$script_path"
      ;;
    plot_*|print_*)
      printf 'plots/%s' "$script_path"
      ;;
    *)
      printf '%s' "$script_path"
      ;;
  esac
}

run() {
  local script_path
  script_path="$(resolve_script_path "$1")"
  if [[ -n "$START_FROM" && "$HAS_STARTED" -eq 0 ]]; then
    if [[ "$script_path" != "$(resolve_script_path "$START_FROM")" ]]; then
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
run "evaluate/evaluate_bundle_checkpoints_on_testsets.py"
run "evaluate/profile_checkpoint_efficiency.py"
run "evaluate/evaluate_cross_dataset_checkpoints.py"

# Downstream analysis/plots.
run "plots/plot_cross_dataset_performance.py"
run "plots/plot_metrics_loss_curves.py"
run "plots/plot_aggregated_loss_curves.py"
run "plots/plot_efficiency_vs_performance.py"
run "plots/plot_pareto_front.py"
run "plots/plot_macro_f1_vs_latency.py"
run "plots/plot_f1_vs_domain_shift.py"
run "plots/plot_test_performance_boxplots.py"
run "plots/print_top_models_by_dataset.py"
run "plots/plot_dataset_sample_panels.py"
run "plots/plot_transform_probe_grid.py"

# Tables / manuscript helpers.
run "tables/generate_test_results_tex_table.py"
run "tables/generate_efficiency_summary_tex_table.py"
run "tables/generate_training_hparams_tex_table.py"
run "tables/generate_inference_efficiency_hparams_tex_table.py"
run "tables/generate_dataset_split_summary_tex_table.py"
run "tables/generate_binary_label_mapping_tex_table.py"
run "tables/generate_augmentations_tex_table.py"

# Generalizability suite
run "generalizability/build_generalization_master_table.py"
run "generalizability/plot_transfer_heatmaps_by_regime.py"
run "generalizability/plot_generalization_gap_and_robustness.py"
run "generalizability/plot_efficiency_vs_generalization.py"
run "generalizability/generate_generalizability_summary_table.py"

# Research-question statistical analyses (outputs under workspace/statistical_results/RQ*)
run "statistics/test_rq1_statistical_analysis.py"
run "statistics/test_rq2_statistical_analysis.py"
run "statistics/test_rq3_statistical_analysis.py"
run "statistics/test_rq4_statistical_analysis.py"
run "statistics/test_rq5_statistical_analysis.py"

if [[ -n "$START_FROM" && "$HAS_STARTED" -eq 0 ]]; then
  echo "[ERROR] START_FROM script not found in pipeline: $START_FROM"
  exit 1
fi

echo
echo "[DONE] All analysis scripts completed."
