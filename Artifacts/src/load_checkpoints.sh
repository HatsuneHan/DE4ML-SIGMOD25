# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Use this script: $0 <experiment_name> <dataset_name> <task>"
  exit 1
fi

EXP_NAME=$1
DATASET_NAME=$2
TASK=$3

# Define the reproduction directory where backups are stored
REPRODUCTION_DIR="../reproduction"

# Define the original data directories to restore files to
DATA_MODELS_DIR="../data/${DATASET_NAME}/${TASK}/datamodels"
MODELS_DIR="../data/${DATASET_NAME}/${TASK}/models"
TMP_DATA_DIR="../data/${DATASET_NAME}/${TASK}/tmp_data"

if [ -d "../data/${DATASET_NAME}/${TASK}" ]; then
  rm -rf "../data/${DATASET_NAME}/${TASK}"
  mkdir -p "../data/${DATASET_NAME}/${TASK}"
else
  mkdir -p "../data/${DATASET_NAME}/${TASK}"
fi

# Check if the reproduction directory exists
if [ -d "../data/${DATASET_NAME}/${TASK}" ]; then

  if [ -d "$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/datamodels" ]; then
    rm -rf "$DATA_MODELS_DIR"
    mkdir -p "$DATA_MODELS_DIR"
  else 
    mkdir -p "$DATA_MODELS_DIR"
  fi
  
  # Restore datamodels.pt files
  if [ -d "$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/datamodels" ]; then
    for iter_dir in "$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/datamodels"/iteration-*; do
      if [ -d "$iter_dir/reg_results" ]; then
        SOURCE_FILE="$iter_dir/reg_results/datamodels.pt"
        
        if [ -f "$SOURCE_FILE" ]; then
          # Get the iteration name, e.g., iteration-1
          ITER_NAME=$(basename "$iter_dir")

          # Create original directory if it doesn't exist
          TARGET_DIR="$DATA_MODELS_DIR/$ITER_NAME/reg_results"
          mkdir -p "$TARGET_DIR"

          # Restore the .pt file
          cp "$SOURCE_FILE" "$TARGET_DIR"
          echo "Restored $SOURCE_FILE to $TARGET_DIR"
        
        else
          echo "Warning: Backup file $SOURCE_FILE does not exist."
        fi
      else
        echo "Warning: Directory $iter_dir/reg_results does not exist."
      fi
    done
  else
    echo "Warning: Backup directory for datamodels does not exist."
  fi

  if [ -d "$MODELS_DIR" ]; then
    rm -rf "$MODELS_DIR"
  fi
  
  # Restore files under models directory
  if [ -d "$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/models" ]; then
    SOURCE_DIR="$REPRODUCTION_DIR/${EXP_NAME}/${DATASET_NAME}/${TASK}/models"
    
    cp -r "$SOURCE_DIR" "../data/${DATASET_NAME}/${TASK}"
    echo "Restored $SOURCE_DIR to ../data/${DATASET_NAME}/${TASK}"
    
  else
    echo "Warning: Backup directory for models does not exist."
  fi

  # if tmp_data not exists, create it
  if [ -d "$TMP_DATA_DIR" ]; then
    rm -rf "$TMP_DATA_DIR"
    mkdir -p "$TMP_DATA_DIR"
  else
    mkdir -p "$TMP_DATA_DIR"
  fi

else
  echo "Warning: Backup directory for experiment $EXP_NAME does not exist."
fi

if [ "${TASK}" = 'enhance' ]; then
  if [ "${DATASET_NAME}" = 'adult_balanced' ]; then
    REPAIRED_DATA_FILE="$REPRODUCTION_DIR/${EXP_NAME}/repaired_data/adult_balanced_repair_15.csv"
  elif [ "${DATASET_NAME}" = 'german' ]; then
    REPAIRED_DATA_FILE="$REPRODUCTION_DIR/${EXP_NAME}/repaired_data/german_repair_21.csv"
  elif [ "${DATASET_NAME}" = 'Bank_balanced' ]; then
    REPAIRED_DATA_FILE="$REPRODUCTION_DIR/${EXP_NAME}/repaired_data/Bank_balanced_repair_17.csv"
  else 
    echo "Warning: No repaired data file found for dataset ${DATASET_NAME}."
  fi

  TARGET_DIR="../data/${DATASET_NAME}/repair/tmp_data"
  if [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"
    mkdir -p "$TARGET_DIR"
  else
    mkdir -p "$TARGET_DIR"
  fi

  if [ -f "$REPAIRED_DATA_FILE" ]; then
    cp "$REPAIRED_DATA_FILE" "$TARGET_DIR"
    echo "Restored $REPAIRED_DATA_FILE to $TARGET_DIR"
  else
    echo "Warning: Repaired data file $REPAIRED_DATA_FILE does not exist."
  fi
else
  echo "Warning: Task ${TASK} is not supported."
fi

echo "Restoration completed."
