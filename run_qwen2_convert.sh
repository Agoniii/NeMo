HF_MODEL='/huangxue/nemo_qwen/Qwen1.5-4B'
NEMO_MODEL='qwen15_4b_mcore.nemo'
python ./scripts/nlp_language_modeling/convert_hf_qwen2_to_nemo.py --in-file ${HF_MODEL} --out-file ${NEMO_MODEL} 2>&1|tee convert.log

