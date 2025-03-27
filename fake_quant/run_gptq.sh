
set -e

export HF_HOME=/mnt/raid0/xuebwang/hf_cache

MODEL_NAME=Llama-3.1-8B # Meta-Llama-3.1-8B # Llama-2-70b-chat-hf # Llama-3.2-1B
MODEL_PATH=/mnt/raid0/pretrained_model/meta-llama/${MODEL_NAME}

HIP_VISIBLE_DEVICES=1 python main.py \
    --model ${MODEL_PATH} \
    --a_bits 16 \
    --v_bits 16 \
    --k_bits 16 \
    --w_bits 4 \
    --lm_eval \
    --tasks wikitext

# --w_rtn:   rtn
# --rotate:  rotate
# --sq:      smoothquant

# python main.py --model /group/amdneuralopt/huggingface/pretrained_models/meta-llama/Meta-Llama-3.1-8B \
#     --a_bits 16 --v_bits 16 --k_bits 16 --w_bits 4  --lm_eval --tasks wikitext --w_rtn
# --rotate --a_bits 16 --v_bits 16 --k_bits 16 --w_bits 4 --w_groupsize 32 --a_groupsize 32 --w_clip
# --rotate --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4  --lm_eval --tasks wikitext 

# Meta-Llama-3.1-8B           w4a16kv16
# fp16                        WIKITEXT2 PPL: 7.33
# mxfp4 only-gptq g=32        WIKITEXT2 PPL: 8.42
# mxfp4 rotate-gptq g=32      WIKITEXT2 PPL: 8.51
# mxfp4 fcsq-rotate-gptq g=32 WIKITEXT2 PPL: 8.54
# mxfp4 sq-rotate-gptq g=32   WIKITEXT2 PPL: 8.51
# mxfp4 rtn                   WIKITEXT2 PPL: 8.35
# nvfp4 rtn                   WIKITEXT2 PPL: 7.90




# Meta-Llama-3.1-8B           w4a4kv16
# fp16                        WIKITEXT2 PPL: 7.33        5.4954  5.5000
# mxfp4 only-gptq g=32        WIKITEXT2 PPL: 8.7959      6.3654
# mxfp4 sq-gptq g=32          WIKITEXT2 PPL: 8.6580  8.6248     6.2875
# mxfp4 rotate-gptq g=32      WIKITEXT2 PPL: 8.9940      6.5367

# mxfp4 sq-rotate-gptq g=32   WIKITEXT2 PPL: 9.6711      6.9245   w R1-R4
# mxfp4 sq-rotate-gptq g=32   WIKITEXT2 PPL: 15.4308     10.6465  w R1-R2


# mxfp4 asq-quark g=32        WIKITEXT2 PPL: 8.9893      6.3654
# mxfp4 awq-quark g=32        WIKITEXT2 PPL: 9.0060      6.3654

# mxfp4 rtn-roundeven         WIKITEXT2 PPL: 9.1717      6.6845
# mxfp4 rtn-roundfloor        WIKITEXT2 PPL: 10.1      

# nvfp4 rtn g=32              WIKITEXT2 PPL: 8.50
# nvfp4 rtn g=16              WIKITEXT2 PPL: 8.26




# Meta-Llama-3.1-8B           w4a4kv16
# fp16                        WIKITEXT2 PPL: 7.33       
# mxfp6 only-gptq g=32        WIKITEXT2 PPL: 7.4226     
# mxfp6 sq-gptq g=32          WIKITEXT2 PPL: 7.4164     
# mxfp6 rotate-gptq g=32      WIKITEXT2 PPL: 7.4364      

# mxfp6 sq-rotate-gptq g=32   WIKITEXT2 PPL: 7.9204       

# mxfp6 asq-quark g=32        WIKITEXT2 PPL: 8.9893     
# mxfp6 awq-quark g=32        WIKITEXT2 PPL: 9.0060     

# mxfp6 rtn-roundeven         WIKITEXT2 PPL: 7.4379      
# mxfp6 rtn-roundfloor        WIKITEXT2 PPL: 7.45      

# nvfp4 rtn g=32              WIKITEXT2 PPL: 8.50
# nvfp4 rtn g=16              WIKITEXT2 PPL: 8.26

