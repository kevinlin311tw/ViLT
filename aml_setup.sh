pip install -r requirements.txt && pip install -e .
ln -s $AML_JOB_INPUT_PATH/t-lqing/vilt_data/official_vilt_data data
ln -s $AML_JOB_INPUT_PATH/t-lqing/experiments
ln -s $AML_JOB_OUTPUT_PATH output
df -h
ls -al
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG=INFO
