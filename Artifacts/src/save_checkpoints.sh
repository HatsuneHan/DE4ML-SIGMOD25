if [ "$#" -ne 3 ]; then
  echo "Use this script: $0 <experiment_name> <dataset_name> <task>"
  exit 1
fi

EXP_NAME=$1
DATASET_NAME=$2
TASK=$3

REPRODUCTION_DIR="../reproduction"
mkdir -p "$REPRODUCTION_DIR"

DATA_MODELS_DIR="../data/${DATASET_NAME}/${TASK}/datamodels"
RESULT_FILE_PATH="../data/${DATASET_NAME}/${TASK}/tmp_data"
MODELS_DIR="../data/${DATASET_NAME}/${TASK}/models"

if [ -d "$DATA_MODELS_DIR" ]; then
  for iter_dir in "$DATA_MODELS_DIR"/iteration-*; do
    if [ -d "$iter_dir/reg_results" ]; then
      SOURCE_FILE="$iter_dir/reg_results/datamodels.pt"

      if [ -f "$SOURCE_FILE" ]; then
        # Get the iteration name, e.g., iteration-1
        ITER_NAME=$(basename "$iter_dir")

        BACKUP_SUBDIR="$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/datamodels/$(basename $iter_dir)/reg_results"
        mkdir -p "$BACKUP_SUBDIR"

        cp "$SOURCE_FILE" "$BACKUP_SUBDIR"
        echo "Saved $SOURCE_FILE to $BACKUP_SUBDIR"
      
      else
        echo "Warning: File $SOURCE_FILE does not exist."
      fi

    else
      echo "Warning: Directory $iter_dir/reg_results does not exist."
    fi
  done

else
  echo "Warning: Directory $DATA_MODELS_DIR does not exist."
fi

echo "Checkpoint saving completed."

# Save the result_dict.json file
if [ -d "$RESULT_FILE_PATH" ]; then
  BACKUP_SUBDIR="$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/result"
  mkdir -p "$BACKUP_SUBDIR"
  cp -r "$RESULT_FILE_PATH" "$BACKUP_SUBDIR"

  echo "Saved $RESULT_FILE_PATH to $BACKUP_SUBDIR"
else
  echo "Warning: File $RESULT_FILE_PATH does not exist."
fi

# Save the models directory
if [ -d "$MODELS_DIR" ]; then
  BACKUP_SUBDIR="$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}"
  mkdir -p "$BACKUP_SUBDIR"

  cp -r "$MODELS_DIR" "$BACKUP_SUBDIR"
  echo "Saved $MODELS_DIR to $BACKUP_SUBDIR"
else
  echo "Warning: Directory $MODELS_DIR does not exist."
fi

echo "Result saving completed."

  
