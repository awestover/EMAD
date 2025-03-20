# In[]:

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

true = []
with open("data/true.txt", "r") as f:
    for l in f.readlines():
        true.append(l)
false = []
with open("data/false.txt", "r") as f:
    for l in f.readlines():
        false.append(l)
NDATA = len(false)
MAXLEN = 75

# device = "cuda"
device = "cpu"
model_name = "gpt2"
assert model_name == "gpt2"
EOS = 50256

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
model = HookedTransformer.from_pretrained(model_name).eval().to(device)


def format(A, B):
    return f"Human: Is either of A or B true? A={A} B={B} Assistant:"


#  lyes = model.forward("number of legs on a dog: 4", return_type="loss").item()
#  lno = model.forward("number of legs on a dog: 3", return_type="loss").item()
#  print(lyes, lno)


def get_layer7(prompt):
    tokens = model.to_tokens(prompt)
    assert len(tokens) <= MAXLEN
    if tokens.shape[1] < MAXLEN:
        padding = torch.full((1, MAXLEN - tokens.shape[1]), 0, device=tokens.device)
        tokens = torch.cat([tokens, padding], dim=1)
    _, cache = model.run_with_cache(tokens)
    return cache["blocks.7.hook_resid_post"]


# In[]
data = []

QUART = NDATA // 4
trues = [true[0:QUART], true[QUART : 2 * QUART]]
falses = [
    false[0:QUART],
    false[QUART : 2 * QUART],
    false[2 * QUART : 3 * QUART],
    false[3 * QUART :],
]

for A, B in zip(falses[0] + trues[0] + falses[1], falses[2] + falses[3] + trues[1]):
    data.append(get_layer7(format(A, B)))

W = torch.cat(data).flatten(1).cpu().numpy()
X = W[: 2 * QUART]
X_OOD = W[2 * QUART :]
Y = torch.cat([torch.zeros(QUART), torch.ones(QUART)])
Y_OOD = torch.ones(QUART)

X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, test_size=0.2)

LRmodel = LogisticRegression(max_iter=1000)
LRmodel.fit(X_train, Y_train)

train_pred = LRmodel.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)
print(train_accuracy)

"""
Q: Does the LRmodel generalize to other sentences of the same type?
"""

eval_pred = LRmodel.predict(X_eval)
eval_accuracy = accuracy_score(Y_eval, eval_pred)
print(eval_accuracy)

"""
A: 
Yeah it works pretty well.
"""

"""
Q: now the real test
does it work for B or A

The hope is that the accuracy is much lower
"""

# In[]
ood_pred = LRmodel.predict(X_OOD)
ood_accuracy = accuracy_score(Y_OOD, ood_pred)
print(ood_accuracy)

# %%
