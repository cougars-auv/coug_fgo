#!/bin/bash
set -e

# --- Selection ---
leaf_dirs=$(cd "${BAGS_DIR}" && find . -name "metadata.yaml" -exec dirname {} \; | sed 's|^\./||')

dir_list="bags\n"
for d in ${leaf_dirs}; do
  p="${d}"
  while [ "${p}" != "." ] && [ "${p}" != "" ]; do
    dir_list="${dir_list}${p}\n"
    p=$(dirname "${p}")
  done
done

selected_dir=$(echo -e "${dir_list}" | sort -u | gum filter --placeholder "Select the directory the study was tuned on ('bags' for all)...") || exit 0
[ -z "${selected_dir}" ] && exit 0

if [ "${selected_dir}" == "bags" ]; then
  study_dir="${BAGS_DIR}"
else
  study_dir="${BAGS_DIR}/${selected_dir}"
fi

# --- Dashboard ---
optuna-dashboard "sqlite:///${study_dir}/optuna_study.db" --host 0.0.0.0 --port 9000
