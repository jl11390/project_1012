#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/scratch/jl11390/%j_%x.out
#SBATCH --error=/scratch/jl11390/%j_%x.err
#SBATCH --account=ds_ga_1012-2023sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jl11390@nyu.edu

# Activate Singularity image
singularity exec --nv --bind /scratch/jl11390/ --overlay /scratch/jl11390/overlay-25GB-500K.ext3:rw /scratch/jl11390/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "source /ext3/env.sh; conda activate; python /home/jl11390/project-1012/counterfactual_model.py"
