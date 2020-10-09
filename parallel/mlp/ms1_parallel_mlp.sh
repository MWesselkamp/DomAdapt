#!/bin/sh 
########## Begin MOAB/Slurm header ##########
#
# Give job a reasonable name
#MOAB -N ms1_CNN
#
# Request number of nodes and CPU cores per node for job
#MOAB -l nodes=3:ppn=20
#
# Estimated wallclock time for job
#MOAB -l walltime=00:07:00:00
#
# Write standard output and errors in same file
#MOAB -j oe 
#
# Send mail when job begins, aborts and ends
#MOAB -m bae
#
########### End MOAB header ##########

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"


# Setup Conda
module load devel/conda/latest

conda activate nnets
python /home/fr/fr_fr/fr_mw263/scripts/ms1_parallel_mlp.py
