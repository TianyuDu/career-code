source /oak/stanford/groups/athey/tianyudu/Development/career_v1_replication/career_v1/bin/activate


# cd /oak/stanford/groups/athey/tianyudu/Development/career_v1_replication/career-code/fairseq
# use the BINARY_DATA_DIR/SAVE_DIR/LOG_DIR on SCRATCH for better disk IO.
cd /scratch/users/tianyudu/career-code/fairseq

ml py-pytorch/1.8.1_py39
ml py-numpy/1.20.3_py39


RESUME_DATA_DIR=/oak/stanford/groups/athey/laborai/Zippia/fairseq_input_unified_one_job_year_without_unemployment_nilf_student_no_skipped_jobs_occ1990dd
du -hs ${RESUME_DATA_DIR}

SURVEY_DATA_DIR=/oak/stanford/groups/athey/career_transformer_data/CAREER_v1_raw_data/PSID79


BINARY_DATA_DIR=/scratch/users/tianyudu/career-code/fairseq/fairseq/data-bin
SAVE_DIR=/scratch/users/tianyudu/career-code/fairseq/fairseq/checkpoints
LOG_DIR=/scratch/users/tianyudu/career-code/fairseq/fairseq/logs
SURVEY_DATASET_NAME=/scratch/users/tianyudu/career-code/fairseq/psid79

mkdir ${BINARY_DATA_DIR}
mkdir ${SAVE_DIR}
mkdir ${LOG_DIR}


sh preprocess/preprocess_transfer_learning_datasets.sh \
  -r $RESUME_DATA_DIR -s $SURVEY_DATA_DIR \
  -b $BINARY_DATA_DIR -n $SURVEY_DATASET_NAME



fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/resume-pretraining \
  --arch career \
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

fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/$SURVEY_DATASET_NAME \
  --finetune-from-model $SAVE_DIR/resume-pretraining/career/checkpoint_best.pt \
  --arch career \
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

cd ../analysis
python compute_survey_data_perplexity.py --model-name career \
  --binary-data-dir $BINARY_DATA_DIR \
  --save-dir $SAVE_DIR \
  --dataset-name $SURVEY_DATASET_NAME

