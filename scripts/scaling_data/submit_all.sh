#!/bin/bash
# Script to submit all PBS scripts in generated_configs directory
# with 1-minute intervals between submissions

CONFIG_DIR="scripts_scaling_data/generated_configs"
LOG_FILE="submission_log_$(date +%Y%m%d_%H%M%S).txt"

# Get all .sh files in the generated_configs directory
scripts=($(ls ${CONFIG_DIR}/*.sh))

echo "Found ${#scripts[@]} PBS scripts to submit"
echo "Submission started at $(date)" | tee -a ${LOG_FILE}
echo "====================================" | tee -a ${LOG_FILE}

# Submit each script with 1-minute interval
for i in "${!scripts[@]}"; do
    script="${scripts[$i]}"
    script_name=$(basename "$script")

    echo "[$(date)] Submitting $script_name (${i+1}/${#scripts[@]})" | tee -a ${LOG_FILE}

    # Submit the PBS script
    job_id=$(qsub "$script" 2>&1)

    if [ $? -eq 0 ]; then
        echo "[$(date)] Successfully submitted: $job_id" | tee -a ${LOG_FILE}
    else
        echo "[$(date)] Failed to submit $script_name: $job_id" | tee -a ${LOG_FILE}
    fi

    # Wait 1 minute before submitting the next script (except for the last one)
    if [ $i -lt $((${#scripts[@]} - 1)) ]; then
        echo "[$(date)] Waiting 1 minute before next submission..." | tee -a ${LOG_FILE}
        sleep 60
    fi
done

echo "====================================" | tee -a ${LOG_FILE}
echo "All submissions completed at $(date)" | tee -a ${LOG_FILE}
echo "Total scripts submitted: ${#scripts[@]}" | tee -a ${LOG_FILE}
echo "Log saved to: ${LOG_FILE}"
