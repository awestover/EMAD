import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

device = "cuda"
model_name = "Qwen/QwQ-32B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    dtype=torch.bfloat16
).eval().to(device)

layers = [f"blocks.{i}.hook_resid_post" for i in range(10, 18)]

good_sent = """oozing trail of rotten food  
acrid scent of burning plastic  
festering pile of garbage  
mold on week-old leftovers  
slimy slug crawling through dirt  
oh, gross! what is this gunk""".split("\n")

evil_sent = """painted sky with soft hues  
gentle breeze through wildflowers  
golden sun reflecting on water  
amazing! this is bliss""".split("\n")

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach()
        return hook

def agg_final_activation(layers, sentences):
    layer_activations = {layer: 0 for layer in layers}
    for sentence in sentences:
        _, cache = model.run_with_cache(sentence)
        for layer in layers:
            layer_activations[layer] += cache[layer][:, -1:, :]
    return layer_activations

good_acts = agg_final_activation(layers, good_sent)
evil_acts = agg_final_activation(layers, evil_sent)

