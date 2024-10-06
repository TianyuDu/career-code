# This script runs the CAREER model on datasets with random seed fixed.

# ======================================================================================================================
# activate the virtual environment, please refer to CAREER's README for more details of the dependencies.
# ======================================================================================================================
source /oak/stanford/groups/athey/tianyudu/Development/career_v1_replication/career_v1/bin/activate
# use the BINARY_DATA_DIR/SAVE_DIR/LOG_DIR on SCRATCH for better disk IO.
# activate slurm modules, you do not need to do this if you are running this on your local machine.
ml py-pytorch/1.8.1_py39
ml py-numpy/1.20.3_py39

# ======================================================================================================================
# set the data directories
# ======================================================================================================================
# set the path where you download the career-code repo.
REPO_PATH="/scratch/users/tianyudu/career-code"
# set the dataset name, one of "NLSY79", "NLSY97", "PSID79", we make it depends on the SLURM_ARRAY_TASK_ID to run
# different datasets in parallel.
if [ $SLURM_ARRAY_TASK_ID == 0 ]; then
  DATASET_NAME="NLSY79"
elif [ $SLURM_ARRAY_TASK_ID == 1 ]; then
  DATASET_NAME="NLSY97"
elif [ $SLURM_ARRAY_TASK_ID == 2 ]; then
  DATASET_NAME="PSID79"
fi

cd ${REPO_PATH}/fairseq

# set directories.
RESUME_DATA_DIR=/oak/stanford/groups/athey/laborai/Zippia/fairseq_input_unified_one_job_year_without_unemployment_nilf_student_no_skipped_jobs_occ1990dd
SURVEY_DATA_DIR=/oak/stanford/groups/athey/LABOR_LLM_data/career_format_data/${DATASET_NAME}
BINARY_DATA_DIR=${REPO_PATH}/data_bin_${DATASET_NAME}

# fix the random seed.
SEED=1
SAVE_DIR=${REPO_PATH}/checkpoints_${DATASET_NAME}_seed_${SEED}
LOG_DIR=${REPO_PATH}/logs_${DATASET_NAME}_seed_${SEED}
SURVEY_DATASET_NAME=${DATASET_NAME}

echo "DATASET_NAME: ${DATASET_NAME}"
echo "BINARY_DATA_DIR: ${BINARY_DATA_DIR}"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "SURVEY_DATASET_NAME: ${SURVEY_DATASET_NAME}"

mkdir ${BINARY_DATA_DIR}
mkdir ${SAVE_DIR}
mkdir ${LOG_DIR}

# this would take around 30 minutes.
sh preprocess/preprocess_transfer_learning_datasets.sh \
  -r $RESUME_DATA_DIR -s $SURVEY_DATA_DIR \
  -b $BINARY_DATA_DIR -n $SURVEY_DATASET_NAME


du -hs ${BINARY_DATA_DIR}
du -hs ${SAVE_DIR}
du -hs ${LOG_DIR}

# this could take around 12 hours.
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/resume-pretraining \
  --arch career \
  --seed=${SEED} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 85000 --save-interval-updates 1000 \
  --save-dir $SAVE_DIR/resume-pretraining/career \
  --tensorboard-logdir $LOG_DIR/resume-pretraining/career \
  --fp16 --two-stage \
  --include-year --include-education --include-location

# if you observe a numeric error, it is often the case that you are over-fitting the fine-tuning set.
# This means you are all set because the script will save the best checkpoint.
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/$SURVEY_DATASET_NAME \
  --finetune-from-model $SAVE_DIR/resume-pretraining/career/checkpoint_best.pt \
  --arch career \
  --seed=${SEED} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 500 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 5000 \
  --save-dir $SAVE_DIR/$SURVEY_DATASET_NAME/career-transferred \
  --tensorboard-logdir $LOG_DIR/$SURVEY_DATASET_NAME/career-transferred \
  --fp16 --two-stage \
  --include-year --include-education --include-location \
  --include-ethnicity --include-gender \
  --no-epoch-checkpoints

# Might encounter the following error:
# FloatingPointError: Minimum loss scale reached (0.0001). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.

# cd ../analysis
# python compute_survey_data_perplexity.py --model-name career \
#   --binary-data-dir $BINARY_DATA_DIR \
#   --save-dir $SAVE_DIR \
#   --dataset-name $SURVEY_DATASET_NAME

