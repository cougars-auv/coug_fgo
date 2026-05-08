#!/bin/bash
set -e

BAG_DIR="${HOME}/cougars-dev/bags"

# --- Selection ---
leaf_dirs=$(cd "${BAG_DIR}" && find . -name "metadata.yaml" -exec dirname {} \; | sed 's|^\./||')

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
  target_dir="${BAG_DIR}"
else
  target_dir="${BAG_DIR}/${selected_dir}"
fi

bags_to_eval=$(find "${target_dir}" -name "metadata.yaml" -exec dirname {} \;)
if [ -z "${bags_to_eval}" ]; then
  echo "No bags found in ${target_dir}"
  exit 0
fi

# --- Options ---
options=$(gum choose --no-limit --header "Select evo options:" -- \
  "--align (Umeyama)" \
  "--align_origin" \
  "--project_to_plane xy ")

evo_base_args=("--t_max_diff" "0.05" "--no_warnings")
[[ "${options}" == *"--align (Umeyama)"* ]] && evo_base_args+=("--align")
[[ "${options}" == *"--align_origin"* ]] && evo_base_args+=("--align_origin")

evo_trans_args=("${evo_base_args[@]}")
evo_rot_args=("${evo_base_args[@]}")
[[ "${options}" == *"--project_to_plane"* ]] && evo_trans_args+=("--project_to_plane" "xy")

# --- Evaluation ---
evo_config set save_traj_in_zip true &>/dev/null

AGENTS=("coug0sim" "coug1sim" "coug2sim" "blue0sim" \
  "bluerov2" "turtlmap" "aquaslam" "aquaslam_wt")
SUFFIXES=("odometry/global" "odometry/global_isam2" "odometry/global_lpi" "odometry/global_tpi" \
  # "odometry/global_iekf" "odometry/global_ukf" "odometry/global_ekf" \
  "odometry/global_tm" "odometry/global_aqs" "odometry/dvl")

for bag_path in ${bags_to_eval}; do
  echo "" && gum style --foreground 75 --bold "Processing ${bag_path}..."

  abs_bag_path=$(cd "${bag_path}" && pwd)

  for agent in "${AGENTS[@]}"; do
    grep -q "name: /${agent}/" "${bag_path}/metadata.yaml" || continue

    # Export ground truth TUM trajectory
    truth="/${agent}/odometry/truth"
    truth_dir="${bag_path}/evo/${agent}/odometry"
    mkdir -p "${truth_dir}"

    if ! ls "${truth_dir}"/*.tum 1> /dev/null 2>&1; then
      echo "" && gum style --foreground 75 --bold "Exporting ground truth for ${agent}..."
      gum spin --spinner dot --title "Exporting GT TUM..." --show-output -- \
        bash -c "cd \"${truth_dir}\" && evo_traj bag2 \"${abs_bag_path}\" \"${truth}\" --save_as_tum"
    fi
    gt_tum=$(ls "${truth_dir}"/*.tum | head -n 1)

    for suffix in "${SUFFIXES[@]}"; do
      topic="/${agent}/${suffix}"
      out_dir="${bag_path}/evo/${agent}/${suffix}"
      if ! ls "${out_dir}"/*.tum 1> /dev/null 2>&1; then
        grep -q "name: ${topic}" "${bag_path}/metadata.yaml" || continue
        grep -A 20 "name: ${topic}" "${bag_path}/metadata.yaml" | grep -m 1 "message_count:" | grep -q "message_count: 0" && continue
      fi
      mkdir -p "${out_dir}"

      if [ -f "${out_dir}/ape_trans.zip" ] && [ -f "${out_dir}/ape_rot.zip" ] && \
         [ -f "${out_dir}/rpe_trans.zip" ] && [ -f "${out_dir}/rpe_rot.zip" ] && \
         ls "${out_dir}"/*.tum 1> /dev/null 2>&1; then
        echo "" && gum style --foreground 245 "Skipping ${topic} (results already exist)"
        continue
      fi

      echo "" && gum style --foreground 75 --bold "Evaluating ${topic}..."

      # Export estimate TUM trajectory
      if ! ls "${out_dir}"/*.tum 1> /dev/null 2>&1; then
        gum spin --spinner dot --title "Exporting TUM..." --show-output -- \
          bash -c "cd \"${out_dir}\" && evo_traj bag2 \"${abs_bag_path}\" \"${topic}\" --save_as_tum"
      fi
      est_tum=$(ls "${out_dir}"/*.tum | head -n 1)

      # APE (Global Accuracy)
      gum spin --spinner dot --title "Calculating APE (Translation)..." --show-output -- \
        evo_ape tum "${gt_tum}" "${est_tum}" -r trans_part "${evo_trans_args[@]}" --save_results "${out_dir}/ape_trans.zip"

      gum spin --spinner dot --title "Calculating APE (Rotation)..." --show-output -- \
        evo_ape tum "${gt_tum}" "${est_tum}" -r angle_deg "${evo_rot_args[@]}" --save_results "${out_dir}/ape_rot.zip"

      # RPE (Drift)
      gum spin --spinner dot --title "Calculating RPE (Translation)..." --show-output -- \
        evo_rpe tum "${gt_tum}" "${est_tum}" -r trans_part "${evo_trans_args[@]}" --delta 1 --delta_unit m --all_pairs --save_results "${out_dir}/rpe_trans.zip"

      gum spin --spinner dot --title "Calculating RPE (Rotation)..." --show-output -- \
        evo_rpe tum "${gt_tum}" "${est_tum}" -r angle_deg "${evo_rot_args[@]}" --delta 1 --delta_unit m --all_pairs --save_results "${out_dir}/rpe_rot.zip"
    done

    if ls "${bag_path}/evo/${agent}"/*/*/*.zip 1> /dev/null 2>&1; then
      echo "" && gum style --foreground 75 --bold "Exporting ${agent} benchmarks..."
      rm -f "${bag_path}/evo/${agent}/benchmark_"*.csv
      for metric in ape_trans ape_rot rpe_trans rpe_rot; do
        if ls "${bag_path}/evo/${agent}"/*/*/${metric}.zip 1> /dev/null 2>&1; then
          evo_res "${bag_path}/evo/${agent}"/*/*/${metric}.zip --save_table "${bag_path}/evo/${agent}/benchmark_${metric}.csv"
        fi
      done
    fi

  done
done

# --- Plots ---
traj_plot_args=("${target_dir}")
[[ "${options}" == *"--align (Umeyama)"* ]] && traj_plot_args+=("--align")
[[ "${options}" == *"--align_origin"* ]] && traj_plot_args+=("--align_origin")
python3 $(dirname "$0")/trajectory_plot.py "${traj_plot_args[@]}"
python3 $(dirname "$0")/timing_plot.py "${target_dir}"
python3 $(dirname "$0")/benchmark_plot.py "${target_dir}"
python3 $(dirname "$0")/lag_plot.py "${target_dir}"
