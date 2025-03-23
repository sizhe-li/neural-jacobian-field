rectory to monitor
DIR_TO_WATCH="/data/scene-rep/u/iyu/scene-jacobian-discovery/data/two_fingers_box_big_rotation_only/train"

# Python script to execute
PYTHON_SCRIPT="/path/to/script.py"

# Check interval in seconds
INTERVAL=1000

while true; do
	# Count the number of files in the directory
	        FILE_COUNT=$(find "$DIR_TO_WATCH" -maxdepth 1 -type f | wc -l)

		    if [[ $FILE_COUNT -ge 256 ]]; then
			            echo "Threshold reached: $FILE_COUNT files found. Running Python script..."
				            CUDA_VISIBLE_DEVICES=7 DISPLAY=:1 python -m jacobian.train dataset.root=/data/scene-rep/u/iyu/scene-jacobian-discovery/data/two_fingers_box_big_rotation_only wandb.name=two_fingers_box_rotation_unet
					            exit 0
						        fi

							    echo "Current file count: $FILE_COUNT. Waiting $INTERVAL seconds..."
							        sleep $INTERVAL
							done
