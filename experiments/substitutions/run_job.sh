#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4      # CPU cores per task
#SBATCH --mem=32G              # Memory
#SBATCH --time=04:00:00        # Max run time (4 hours)
#SBATCH --output=slurm_out.out
#SBATCH --error=slurm_out.err
#SBATCH --account=aip-gpekhime

# Load modules
# module load python/3.10
# module load cuda/12.1 # Adjust version as needed for Killarney

# # Create/Activate Virtual Env
# if [ ! -d "venv_killarney" ]; then
#     echo "Creating virtual environment..."
#     python -m venv venv_killarney
#     source venv_killarney/bin/activate
#     pip install --upgrade pip
#     pip install vllm datasets spacy python-dotenv nltk
#     python -m spacy download en_core_web_sm
    
#     # Install NLTK data if needed
#     python -c "import nltk; nltk.download('wordnet')"
# else
#     source venv_killarney/bin/activate
# fi

# Run Evaluation
# Modify this line to point to the specific experiment script you want to run
echo "Starting Evaluation..."

# Set HuggingFace Cache to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/6107991/golikovp/cache
mkdir -p $HF_HOME

# Example: Run Opposi# Run the consolidated baseline evaluation
# Default model is GAIR/LIMO, but can be overridden with --model if needed
python evaluate.py --limit 1

echo "Job Complete"
