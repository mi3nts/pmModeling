#!/bin/bash
#SBATCH -J pmEstimation         # Job name
#SBATCH -o /scratch/prabuddha/2yrHist_train/errFiles/pmEstimation.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -e /scratch/prabuddha/2yrHist_train/errFiles/pmEstimation.%j.err  # Error File Name 
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                   # Total number of mpi tasks requested
#SBATCH -t 05:00:00             # Run time (hh:mm:ss) - 24 hours

export MODULEPATH="/opt/ohpc/pub/unpackaged/modulefiles:$MODULEPATH"
ml miniconda
source activate pmtest
python3 pythonSyncher.py mintsDefinitionsBirdCall.yaml
