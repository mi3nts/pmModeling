#!/bin/bash
#SBATCH -J dixie_fire         # Job name
#SBATCH -o /scratch/prabuddha/dust_events/errFiles/dixie_fire.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -e /scratch/prabuddha/dust_events/errFiles/dixie_fire.%j.err  # Error File Name 
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                   # Total number of mpi tasks requested
#SBATCH -t 12:00:00             # Run time (hh:mm:ss) - 24 hours
export MODULEPATH="/opt/ohpc/pub/unpackaged/modulefiles:$MODULEPATH"
ml miniconda

source activate pmtest
#python3 pm_est.py
#python3 data_process.py
python3 main.py
