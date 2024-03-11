import os
import torch
import argparse
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained("/huangxue/nemo_qwen/Qwen1.5-4B", trust_remote_code=True)

model = Qwen2ForCausalLM.from_pretrained("/huangxue/nemo_qwen/Qwen1.5-4B", trust_remote_code=True)
#for name, param in model.named_parameters():
#    print(f"1111 - {name}", param.shape)
#    if 'layernorm' in name:
#        print(param)
model = model.cuda()

model = model.eval()
with torch.no_grad():
    #text = ["This is a piece of text.", "Another piece of text.", "today is a nice day!"]
    text = ["today is a nice day."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    encoded_input = {k:v.cuda() for k, v in encoded_input.items()}
    print("encoded_input", encoded_input, encoded_input["input_ids"].dtype)
    dst_output = model(**encoded_input)
    print("dst_output[logits]", dst_output["logits"].shape, dst_output["logits"].dtype, dst_output["logits"])

# encoded_input {'input_ids': tensor([[30113,   374,   264,  6419,  1899,    13]], device='cuda:0'), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]], device='cuda:0')} torch.int64
# dst_output[logits] torch.Size([1, 6, 151936]) torch.float32 
# tensor([[[ 4.8704,  5.2913,  0.3935,  ..., -8.2262, -8.2250, -8.2275],
#          [ 2.1064,  5.3246,  0.4860,  ..., -8.0952, -8.0962, -8.0973],
#          [-0.9267,  2.5504, -1.9470,  ..., -8.2106, -8.2108, -8.2121],
#          [ 4.9554,  4.0781, -0.0871,  ..., -7.0798, -7.0796, -7.0815],
#          [10.9436,  5.5378,  1.5921,  ..., -7.5402, -7.5392, -7.5405],
#          [ 2.4866, -3.6550, -2.3241,  ..., -6.3707, -6.3670, -6.3694]]],
#        device='cuda:0')

# nemo
# 'token_ids': [[30113, 374, 264, 6419, 1899, 13]]
# output:  torch.Size([1, 6, 151936]) torch.float32 
# tensor([[[ 4.8700,  5.2901,  0.3908,  ..., -8.2265, -8.2253, -8.2278],
#          [ 2.1008,  5.3301,  0.4936,  ..., -8.0948, -8.0958, -8.0969],
#          [-0.9318,  2.5442, -1.9493,  ..., -8.2140, -8.2142, -8.2155],
#          [ 4.9552,  4.0743, -0.0882,  ..., -7.0819, -7.0817, -7.0836],
#          [10.9436,  5.5374,  1.5926,  ..., -7.5387, -7.5377, -7.5390],
#          [ 2.4885, -3.6559, -2.3248,  ..., -6.3695, -6.3658, -6.3682]]],
#        device='cuda:0')

# tp2pp2
# output:  torch.Size([1, 6, 151936]) torch.float32 tensor([[[ 4.8700,  5.2901,  0.3908,  ..., -8.2265, -8.2253, -8.2278],
#          [ 2.1007,  5.3301,  0.4936,  ..., -8.0948, -8.0958, -8.0969],
#          [-0.9318,  2.5442, -1.9493,  ..., -8.2140, -8.2142, -8.2155],
#          [ 4.9552,  4.0743, -0.0882,  ..., -7.0819, -7.0817, -7.0836],
#          [10.9436,  5.5374,  1.5926,  ..., -7.5387, -7.5377, -7.5390],
#          [ 2.4885, -3.6559, -2.3248,  ..., -6.3695, -6.3658, -6.3682]]],
#        device='cuda:2')