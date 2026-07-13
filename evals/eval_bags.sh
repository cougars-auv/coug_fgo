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

# --- Options ---
options=$(gum choose --no-limit --header "Select evo options:" -- \
  "--align (Umeyama)" \
  "--project_to_plane xy")

eval_flags=()
[[ "${options}" == *"--align (Umeyama)"* ]] && eval_flags+=("--align")
[[ "${options}" == *"--project_to_plane"* ]] && eval_flags+=("--project_to_plane")

# --- Evaluation ---
python3 "$(dirname "$0")/bag_evaluator.py" "${target_dir}" "${eval_flags[@]}"
