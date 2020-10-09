#!/bin/bash
########## Begin Slurm header ##########
#
# Give job a reasonable name
#SBATCH --job-name=hparams_search_convnet
#
#SBATCH --ntasks=50
#
# Estimated wallclock time for job
#SBATCH --time=06:00:00
#
# RAM
#SBATCH --mem=6400mb
#
# Send mail when job begins, aborts and ends
#SBATCH --mail-type=END
#
#SBATCH --mail-user=marieke.wesselkamp@posteo.de
#
#SBATCH --output=out_conv
#SBATCH --error=errors
#
########### End header ##########

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $SLURM_JOBID"
echo "Number of cores allocated to job:     ${SLURM_JOB_CPUS_PER_NODE}"

# Setup Conda
module load devel/miniconda
conda activate nnets

python /pfs/data5/home/fr/fr_fr/fr_mw263/ms_parallel_conv.py

conda deactivate