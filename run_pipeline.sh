#!/bin/bash
# ============================================================
# IR Annotation Pipeline — Master Orchestrator
# ============================================================
# Usage:
#   bash run_pipeline.sh [dataset_path] [output_dir] [num_gpus]
#
# Runs 4 sequential phases. Each phase reads the output of
# the previous one. Phases are checkpointed — if a phase
# fails, re-run this script and it will resume from the last
# successful phase.
#
# Requirements:
#   - data-juicer installed  (pip install py-data-juicer)
#   - CUDA available
# ============================================================

DATASET_PATH="${1:-/data/robot_dataset}"
OUTPUT_DIR="${2:-/data/ir_annotations}"
NUM_GPUS="${3:-4}"

mkdir -p "$OUTPUT_DIR"
CHECKPOINT_FILE="$OUTPUT_DIR/.phase_checkpoint"

# ── helpers ────────────────────────────────────────────────────
phase_done() {
    grep -q "phase$1" "$CHECKPOINT_FILE" 2>/dev/null
}

mark_done() {
    echo "phase$1" >> "$CHECKPOINT_FILE"
    echo "[pipeline] Phase $1 complete ✓"
}

run_dj() {
    local phase=$1
    local yaml=$2
    local input=$3
    local output=$4

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Phase $phase — $(basename $yaml)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Patch dataset_path and export_path into the yaml at runtime
    PATCHED_YAML="/tmp/phase${phase}_patched.yaml"
    sed -e "s|dataset_path:.*|dataset_path: '$input'|" \
        -e "s|export_path:.*|export_path: '$output'|" \
        "$yaml" > "$PATCHED_YAML"

    python -m data_juicer.tools.process_data \
        --config "$PATCHED_YAML" \
        --np "$NUM_GPUS"

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[pipeline] Phase $phase FAILED (exit $exit_code)"
        exit $exit_code
    fi
}

# ── Phase 1: Foundation ────────────────────────────────────────
if ! phase_done 1; then
    run_dj 1 \
        "phase1_foundation.yaml" \
        "$DATASET_PATH" \
        "$OUTPUT_DIR/phase1_out.jsonl"
    mark_done 1
else
    echo "[pipeline] Phase 1 already done, skipping"
fi

# ── Phase 2: Geometry & Projection ────────────────────────────
if ! phase_done 2; then
    run_dj 2 \
        "phase2_geometry.yaml" \
        "$OUTPUT_DIR/phase1_out.jsonl" \
        "$OUTPUT_DIR/phase2_out.jsonl"
    mark_done 2
else
    echo "[pipeline] Phase 2 already done, skipping"
fi

# ── Phase 3: Composite IR ─────────────────────────────────────
if ! phase_done 3; then
    run_dj 3 \
        "phase3_composite.yaml" \
        "$OUTPUT_DIR/phase2_out.jsonl" \
        "$OUTPUT_DIR/phase3_out.jsonl"
    mark_done 3
else
    echo "[pipeline] Phase 3 already done, skipping"
fi

# ── Phase 4: Goal-State IR ────────────────────────────────────
if ! phase_done 4; then
    run_dj 4 \
        "phase4_goalstate.yaml" \
        "$OUTPUT_DIR/phase3_out.jsonl" \
        "$OUTPUT_DIR/phase4_out.jsonl"
    mark_done 4
else
    echo "[pipeline] Phase 4 already done, skipping"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " All phases complete."
echo " Final output: $OUTPUT_DIR/phase4_out.jsonl"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
