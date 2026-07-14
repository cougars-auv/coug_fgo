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

selected_dir=$(echo -e "${dir_list}" | sort -u | gum filter --placeholder "Select directory or bag to evaluate ('bags' for all)..." || exit 0)
[ -z "${selected_dir}" ] && exit 0

if [ "${selected_dir}" == "bags" ]; then
  target_dir="${BAGS_DIR}"
else
  target_dir="${BAGS_DIR}/${selected_dir}"
fi

agents=$(basename -a "${CONFIG_DIR}"/*_params.yaml | sed 's/_params.yaml$//' | sort | gum choose --no-limit --header "Select agents to evaluate..." || exit 0)
[ -z "${agents}" ] && exit 0

# --- Options ---
evo_options=$(gum choose --no-limit --header "Select evo flags:" -- "--align" "--project_to_plane xy") || exit 0
evo_flags=$(echo "${evo_options}" | tr '\n' ' ')

# --- Evaluate ---
python3 "$(dirname "$0")/_eval_bags.py" --target-dir "${target_dir}" --agents ${agents} --evo-flags="${evo_flags}"
