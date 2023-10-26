#!/bin/sh

# .Load required modules. The modules can be loaded either from the user's locally installed software or provided by the platform's installed modules. We strongly recommend opting for the latter choice.

module purge
module load cuda/11.0.1
module load gcc/7.5.0
module load gcc/11.2.0
module load cmake/3.18.2

export PSP_OPENIB=1
export PSP_UCP=1
export PSP_CUDA=1

#export DATA=/path/to/data
export DATA=/p/project/joaiml/hetgrad/data
export CUDA_VISIBLE_DEVICES="0,1,2,3"


# .Parameter configuration. The following parameters are required: 
#1) Number of nodes; 2) hardware partition (maybe not required); 3) training mode (proposed); 4) batch_size; 5) number of epochs; 6) dataset (CIFAR10, CIFAR100, MNIST or IMAGENET); 7) K' hyperparameter; 8) model (r18, r34, r50, r101, shufflenet_v2_1x, vit_timm, vit, vit_small); 9) learning rate; 10) optimizer

# .Parse named parameters using getopts
while getopts ":n:p:m:b:e:d:s:o:l:r:z:w:x:" opt; do
    case $opt in
        n) numNodes="$OPTARG";;
        p) partition="$OPTARG";;
        m) mode="$OPTARG";;
        b) bsz="$OPTARG";;
        e) epochs="$OPTARG";;
        d) dt="$OPTARG";;
        s) sharing="$OPTARG";;
        o) model="$OPTARG";;
        l) lr="$OPTARG";;
        r) optim="$OPTARG";;
        z) seed="$OPTARG";;
        w) wd="$OPTARG";;
        x) experiment="$OPTARG";;
        \?) echo "Invalid option: -$OPTARG" >&2;;
    esac
done

# .Shift the parameters so the remaining positional arguments are accessible
shift "$((OPTIND - 1))"

#CONFIG_FOLDER=/example/url/to/config
CONFIG_FOLDER=/home/smoreno/ColossalAI/fpm$numNodes
HOSTFILE=$CONFIG_FOLDER/hosts
scontrol show hostname ${SLURM_JOB_NODELIST} | sort > $HOSTFILE
RANKFILE=$CONFIG_FOLDER/rnk_file_conf
CFILE=$CONFIG_FOLDER/M"$numNodes".conf
echo -e "# Configuration: M"$numNodes  > $CFILE

# .Configure node hardware devices. This is employed for the generation of an MPI rank_file intended for execution purposes. In the subsequent illustration, we establish two distinct node configurations, each with allocations of 2 GPUs and 4 GPUs, respectively. Within this context, the identifier denoted as "NODE" signifies the node's nomenclature. Subsequently, the succeeding pair of integers designates both the process ordinal number within the node and the predetermined core designated for GPU-CPU communication. Meanwhile, the "id" value corresponds to the identifier associated with the GPU device.


for line in $(cat $HOSTFILE)
do
	# .Write the hostfile name to control number of processes per node regarding available GPUs.
	node_name[$i]="$line"
	node_info=$(scontrol show node="$line" | grep "gpu:$partition:")
	node_gpus=$(echo "$node_info" | awk -F'[:=,]' -v partition="$partition" '{for(i=1; i<=NF; i++) if ($i == partition) print $(i+1)}')

	
	#node_gpus=$((10#${node_info: -1} -1))
	let "i += 1"
	
	# .Write the .conf file
	cnf_gpunode=()
	p=0
	for p in $(seq 0 $(($node_gpus - 1)));
	do
		echo $p
		cnf_gpunode+=("NODE $p $((p+1)) gpu id=$p")
		c_line=${cnf_gpunode[$p]//NODE/$line}
		echo -e "$c_line" >> $CFILE
	done
done


# .Header of CFILE and RANKFILE: from this file we generate the RANKFILE for MPI
echo -e "# MPI RANKFILE: $RANKFILE"  > $RANKFILE


# .From each line of the Mx.conf file we create a line in each one of the CONFFILE and RANKFILE files, with the correct format.
rank_nr=0
while IFS='' read -r line || [[ -n "$line" ]]; do

    new_line=( $line )

    # Skip or copy the comment lines
    carac=${line:0:1}
    if [ "$carac" == "#" ]; then
        echo $line >> $RANKFILE
    elif [ ! -z "$line" ]
    then
        rank_line="rank $rank_nr=${new_line[0]} slot=${new_line[2]}"
        let "rank_nr += 1"
        # Copy to files
        echo -e $rank_line >> $RANKFILE
    fi
done < $CFILE
numProcs=$rank_nr

# .Execute the algorithm for 5 MonteCarlo runs. 
for i in 5
do
#     sbatch --nodes 2 -p volta --gpus-per-node=2 -t 02:00:00 ./launch_github.sh -n 2 -p volta -m proposed -d CIFAR10 -s 50 -o resnet18
    if [ "$experiment" == "CIFAR" ]; then
        ~/mpi_install/bin/mpirun -n $numProcs --rankfile $RANKFILE --mca btl tcp,self,vader --mca pml ob1 -report-bindings --display-map --bind-to core --oversubscribe -quiet python ./CIFAR/main.py --manualSeed $seed --bsz 128 --epochs 400 --mode $mode --worldsize $numProcs --dataset $dt --model $model --lr 0.1 --wd 5e-4 --sharingiter -1
#     sbatch --nodes 2 -p volta --gpus-per-node=2 -t 02:00:00 ./launch_github.sh -n 2 -p volta -m proposed -x GAN -z 3333
    elif [ "$experiment" == "GAN" ]; then
        ~/mpi_install/bin/mpirun -n $numProcs --rankfile $RANKFILE --mca btl tcp,self,vader --mca pml ob1 -report-bindings --display-map --bind-to core --oversubscribe -quiet python ./GAN/gan.py --mode $mode --manualSeed $seed --worldsize $numProcs
    elif [ "$experiment" == "FINEGRAINED" ]; then
        ~/mpi_install/bin/mpirun -n $numProcs --rankfile $RANKFILE --mca btl tcp,self,vader --mca pml ob1 -report-bindings --display-map --bind-to core --oversubscribe -quiet python ./FINEGRAINED/main.py --manualSeed $seed --epochs 100 --mode $mode --worldsize $numProcs --dataset $dt --model $model --lr 0.1 --wd 5e-4 --sharingiter -1
    elif [ "$experiment" == "IMAGENET" ]; then
        ~/mpi_install/bin/mpirun -n $numProcs --rankfile $RANKFILE --mca btl tcp,self,vader --mca pml ob1 -report-bindings --display-map --bind-to core --oversubscribe -quiet python main.py --manualSeed $seed --bsz $bsz --epochs $epochs --mode $mode --worldsize $numProcs --dataset $dt --model $model --optim $optim --lr $lr --wd $wd --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decouple True --when 70 80 --partition $partition --sharingiter $sharing
    fi
done


