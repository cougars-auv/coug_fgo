#!/bin/bash
set -e

# --- Selection ---
while true; do
  bags=$(cd "${BAGS_DIR}" && find . -name "metadata.yaml" -exec dirname {} \; | sed 's|^\./||' | sort -r | gum choose --no-limit --header "Select bags to process offline...") || exit 0
  [ -n "${bags}" ] && break
done

namespace=$(basename -a "${CONFIG_DIR}"/*_params.yaml | sed 's/_params.yaml$//' | sort | gum filter --placeholder "Select an agent namespace...") || exit 0
[ -z "${namespace}" ] && exit 0

# --- Options ---
evo_options=$(gum choose --no-limit --header "Select evo flags:" -- "--align" "--project_to_plane xy") || exit 0
evo_flags=$(echo "${evo_options}" | tr '\n' ' ')

bag_paths=()
for b in ${bags}; do
  bag_paths+=("${BAGS_DIR}/${b}")
done

# --- Process ---
python3 "$(dirname "$0")/_run_offline_fgo.py" --namespace "${namespace}" --bags "${bag_paths[@]}" --evo-flags="${evo_flags}"
