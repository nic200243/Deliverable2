# Deliverable2

The programs used for the lab report are d2OMP.cpp and d2MPI.cpp
d2OMP.cpp contains implementations of sequential and OpenMP functions and the code to run them changing matrix size and thread count.
d2MPI.cpp contains implementations of MPI functions and the code to run them changing matrix size.

In order to reproduce the results for sequential and OpenMP functions:
-Connect to the Unitn HPC Cluster
-copy d2OMP.cpp and PBSomp.pbs in a folder
-open PBSomp.pbs and change the working directory
-use qsub PBSomp.pbs to start the job
-an OutputOMP.o file will be generated and it will contain the time measurements. 

In order to reproduce the results for MPI functions:
-Connect to the Unitn HPC Cluster
-copy d2MPI.cpp and PBSmpi.pbs in a folder
-open PBSmpi.pbs and change the working directory
-use qsub PBSmpi.pbs to start the job
-an OutputMPI.o file will be generated and it will contain the time measurements. 

outputMPI.o and OutputOMP.o files in this repository contain the time measurements used for the lab report
