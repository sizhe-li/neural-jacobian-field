#!/bin/bash

# Function to check if a GPU has enough free memory
check_gpu_memory() {
    local gpu_id=$1
    local required_memory=25000
    local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${gpu_id})
    if (( free_memory >= required_memory )); then
        return 0
    else
        return 1
    fi
}

# Check if the model name argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo "model_name should be either 'transformer' or 'unet'"
    exit 1
fi

model_name=$1

# Define the range of consistency_loss_weight values for the grid search
weights=(0.0 0.5 1.0 2.0 4.0)

gpu_counter=0

# Loop over each weight value and run the training command on available GPUs
for i in $(seq 0 $((${#weights[@]} - 1))); do
    weight=${weights[$i]}
    if check_gpu_memory ${gpu_counter}; then
        echo "Running training with consistency_loss_weight=${weight} on GPU ${gpu_counter}"
        CUDA_VISIBLE_DEVICES=${gpu_counter} DISPLAY=:1 python -m jacobian.train \
            dataset.root=/home/iyu/scene-jacobian-discovery/data/two_fingers_composition_${model_name} \
            wandb.name=two_finger_${model_name}_${weight} \
            wrapper.model.name=${model_name} \
            wrapper.model.consistency_loss_weight=${weight} &
        ((gpu_counter++))
    else
        echo "Skipping GPU ${gpu_counter} due to insufficient memory"
        ((gpu_counter++))
    fi
done

# Wait for all background processes to finish
wait