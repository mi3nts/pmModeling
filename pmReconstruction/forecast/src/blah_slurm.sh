#!/bin/bash
#SBATCH -J pmFC_est         # Job name
#SBATCH -o /scratch/prabuddha/pm_est_fc/errFile/pmEstimation.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -e /scratch/prabuddha/pm_est_fc/errFile/pmEstimation.%j.err  # Error File Name 
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                   # Total number of mpi tasks requested
##SBATCH --array=0-19            # Array ranks to run
#SBATCH -t 12:00:00             # Run time (hh:mm:ss) - 24 hours

export MODULEPATH="/opt/ohpc/pub/unpackaged/modulefiles:$MODULEPATH"
ml miniconda
source activate pmtest

python3 blah.py
