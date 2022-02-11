#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/configs"
servername="localhost"
workspace="workspaces/poc_workspace"
admin_username="admin"  # default admin
site_pre="site-"

n_clients=$1
n_total=$2
config=$3
run=$4
alpha=$5

if test -z "${n_clients}" || test -z "${n_total}" || test -z "${config}" || test -z "${run}" || test -z "${alpha}"
then
      echo "Usage: ./run_poc.sh [n_clients] [n_total] [config] [run] [alpha], e.g. ./run_poc.sh 8 64 cifar10_fedavg 1 0.1"
      exit 1
fi

n_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "There are ${n_gpus} GPUs."

echo "RUNNING ${n_total} VIRTUAL CLIENTS ON ${n_clients} REAL ONES..."

# start server
echo "STARTING SERVER"
export CUDA_VISIBLE_DEVICES=0
./${workspace}/server/startup/start.sh ${servername} &
sleep 10

# start clients
echo "STARTING ${n_clients} REAL CLIENTS"
for id in $(eval echo "{1..$n_clients}")
do
  gpu_idx=$((${id} % ${n_gpus}))
  echo "Starting client${id} on GPU ${gpu_idx}"
  export CUDA_VISIBLE_DEVICES=${gpu_idx}
  ./${workspace}/"${site_pre}${id}"/startup/start.sh ${servername} "${site_pre}${id}" &
done
sleep 30

# only split the data if simulating more than one client, not needed for simulating central training with 1 client
if [ "${n_clients}" -gt 1 ]
then
  # download and split data
  echo "PREPARING DATA"
  rm "/tmp/cifar10_data/*.npy"  # remove old splits
  python3 ./pt/utils/prepare_data.py --data_dir="/tmp/cifar10_data" --num_sites="${n_total}" --alpha="${alpha}"
fi

# start training
echo "STARTING TRAINING"
python3 ./run_fl.py --port=8003 --admin_dir="./${workspace}/${admin_username}" \
  --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${n_clients}" --poc

# sleep for FL system to shut down, so a new run can be started automatically
sleep 30
