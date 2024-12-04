#!/bin/bash
#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --constraint=a100_40
#SBATCH --cpus-per-task=32

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Print system information
echo "IDs of GPUs available: $CUDA_VISIBLE_DEVICES"
echo "No of GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "No of CPUs available: $SLURM_CPUS_PER_TASK"
echo "nproc output: $(nproc)"
nvidia-smi

# Generate unique Job ID
if [ "$SLURM_ARRAY_JOB_ID" ]; then
    job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    job_id="$SLURM_JOB_ID"
fi

# Project configuration
repo="fr_bias"
project_dir="/bask/projects/v/vjgo8416-tdi4/"
work_dir="$project_dir/${repo}"
st_folder="ST1"  # Change this for different runs: ST1, ST2, etc.

# List of scripts to run with their subdirectories
scripts=(
    "skin_analysis/run_deeplab.py"
    "skin_analysis/extract_skin_scores.py"
    # Add more scripts here as needed
)

# Module loading
module purge
module load baskerville
module load Miniconda3/4.10.3

set -e  # Exit on error

# Initialize conda
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Conda environment setup
export CONDA_ENV_PATH="/bask/projects/v/vjgo8416-tdi4/conda_envs/${repo}"
export CONDA_PKGS_DIRS="/tmp"

# Create and activate conda environment if it doesn't exist
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Creating new conda environment at ${CONDA_ENV_PATH}"
    conda create --yes --name ${repo} --prefix ${CONDA_ENV_PATH} python=3.10
    conda activate ${CONDA_ENV_PATH}
    
    # Install requirements from file
    echo "Installing requirements from requirements.txt"
    pip install -r ${work_dir}/requirements.txt
else
    echo "Activating existing conda environment at ${CONDA_ENV_PATH}"
    conda activate ${CONDA_ENV_PATH}
fi

# Check PyTorch GPU access
echo "Checking PyTorch GPU availability..."
python -c "import torch; print('PyTorch GPU available:', torch.cuda.is_available()); print('GPU device count:', torch.cuda.device_count()); print('GPU device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Record start time
start_time=$(date -Is --utc)

# Change to working directory
cd ${work_dir}

# Execute each script in sequence
for script in "${scripts[@]}"; do
    script_dir=$(dirname "$script")
    script_name=$(basename "$script")
    
    # Record start time for this script
    script_start_time=$(date -Is --utc)
    echo "Starting $script_name at $script_start_time"
    
    # Change to script's directory and run
    cd "${work_dir}/${script_dir}"
    if [[ "$script_name" == "extract_skin_scores.py" ]]; then
        python "${script_name}" ${st_folder}
    else
        python "${script_name}"
    fi
    
    # Return to main working directory
    cd "${work_dir}"
    
    # Record end time for this script
    script_end_time=$(date -Is --utc)
    echo "Finished $script_name"
    echo "Script timing:"
    echo "  started: $script_start_time"
    echo "  finished: $script_end_time"
    echo "----------------------------------------"
done

# Record end time
end_time=$(date -Is --utc)

# Print summary
echo "All jobs completed"
echo "started: $start_time"
echo "finished: $end_time"

