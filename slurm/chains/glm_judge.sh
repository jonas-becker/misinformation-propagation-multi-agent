#!/usr/bin/env bash
# GLM-4.7-Flash judge pass on MINT (second judge). Separate job so it can be ordered after the GPT-OSS judge pass
# (both edit MINT_dataset_v1.2 in place). Needed before any --mint_filter verified run.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
judge MINT_dataset_v1.2
step "validation tables" python scripts/validate_mint.py --stats_only
finish
