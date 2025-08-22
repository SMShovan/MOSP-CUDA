# Instructions to run the code

## SYCL Project

### Hellblender cluster dependency 

ssh \<toYourMachine\>

srun -p gpu --gres gpu:A100:1 -N 1 --ntasks-per-node 8 -t 02:00:00 --mem 200G --pty /bin/bash

module avail 
module load cuda/11.8.0_gcc_9.5.0
module load cmake/3.26.3_gcc_9.5.0
module load miniconda3
conda create -n sycl_env  (environment location: /home/akkcm/.conda/envs/sycl_env)
source activate sycl_env
python -m pip install ninja
export DPCPP_HOME=~/sycl_workspace
mkdir $DPCPP_HOME
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
python $DPCPP_HOME/llvm/buildbot/configure.py --cudaexit
python $DPCPP_HOME/llvm/buildbot/compile.py

Every time after openinng new window run below commands:
export DPCPP_HOME=~/sycl_workspace && export PATH=$DPCPP_HOME/llvm/build/bin:$PATH && export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH && cd sycl_workspace/GPUMultiObjective/tools/


### Building the SYCL project and run
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda SYCL_Final.cpp -o SYCL_Final && ./SYCL_Final

## OpenMP project

g++ -fopenmp -std=c++11 -o program main.cpp

## Base paper

### Clone the library
git clone git@github.com:SMShovan/multicrit.git

### Source the libraries 
source ../lib/tbb/bin/tbbvars.sh intel64
### reflect changes
make configure
### make 
make all
### run scripts
From scripts/ 
run build_binaries.sh

