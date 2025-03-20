#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install transformer_lens')


# In[1]:


false = """The sun sets in the north.
Water boils at 10°C.
Humans have three eyes.
The Earth is flat.
Fish live in the sky.
Fire is cold.
Ice is hot.
The moon is made of cheese.
Birds have fur.
Plants grow without water or sunlight.
A cat is a reptile.
The sky is always purple.
Gravity makes objects float upwards.
Humans breathe helium.
A year has 500 days.
The ocean is filled with chocolate.
Dogs meow.
Apples grow underground.
The human heart is located in the leg.
A triangle has four sides.
The Earth is made of cheese.
A car has wings.
Snow is black.
The brain is located in the foot.
The alphabet has 100 letters in English.
The Pacific Ocean is the smallest ocean.
The Eiffel Tower is in New York.
Penguins can fly.
Milk comes from fish.
The color of grass is blue.
There are 50 months in a year.
Giraffes have short necks.
Humans cannot taste food.
Fish can walk on land.
Tigers have polka dots.
The sun is a planet.
Frogs cannot jump.
A clock tells you the weather.
The Earth has 10 moons.
A bicycle has 10 wheels.
Lightning happens only on sunny days.
Trees are made of metal.
Butterflies cannot fly.
Elephants are the smallest animals.
Kangaroos swim instead of hopping.
Water is made of gold and silver.
Bats are active during the day.
Rainbows are made of sound.
Humans do not have bones.
A bee cannot fly.
Humans can breathe underwater.
The Moon is made of cheese.
Dogs can speak English.
Elephants can fly.
The Earth is flat.
Fish can live on land.
Penguins live at the North Pole.
Cars run on water.
Trees can walk.
The Sun revolves around the Earth.
Cats and dogs are the same species.
Humans have three eyes.
The sky is green.
Rocks are alive.
Bananas are blue.
Chickens have teeth.
Books can read themselves.
Grass is red.
Paris is the capital of Spain.
Whales are fish.
The Earth has two suns.
Oranges grow underground.
Fire is cold.
Metal floats on water.
Giraffes have short necks.
Humans can photosynthesize.
Cows can fly.
Water boils at 10 degrees Celsius.
Pencils are made of rubber.
The Sahara is covered in ice.
Bicycles have three wheels.
Water flows uphill naturally.
Lions are herbivores.
Humans can see radio waves.
Birds are reptiles.
Snow is hot.
The Earth is the largest planet.
Diamonds are soft.
Grass grows at night only.
Carrots help you see in the dark.
Bees make milk.
Wood is magnetic.
Watches have square faces.
Kangaroos are birds.
Eggs are square.
Mice are bigger than cats.
Humans have tails.
The Atlantic Ocean is the smallest ocean.
Apples grow underground.
Chocolate grows on trees.""".split("\n")


# In[2]:


true = """The sun rises in the east.
Water freezes at 0°C.
Humans have two eyes.
The Earth revolves around the sun.
Fish live in water.
Fire is hot.
Ice is cold.
The moon orbits the Earth.
Birds have feathers.
Plants need sunlight to grow.
A cat is a mammal.
The sky is often blue.
Gravity pulls objects downwards.
Humans breathe oxygen.
A year has 365 days.
The ocean contains saltwater.
Dogs bark.
Apples grow on trees.
The human heart pumps blood.
A triangle has three sides.
The Earth is a planet.
A car has wheels.
Snow is white.
The brain controls the body.
The alphabet has 26 letters in English.
The Pacific Ocean is the largest ocean.
The Eiffel Tower is in Paris.
Penguins cannot fly.
Milk comes from cows.
The color of grass is green.
There are 12 months in a year.
Giraffes have long necks.
Humans can taste food.
Some fish have scales.
Tigers have stripes.
The sun is a star.
Frogs can jump.
A clock tells time.
The Earth has one moon.
A bicycle has two wheels.
Lightning occurs during thunderstorms.
Trees provide shade.
Butterflies have wings.
Elephants are the largest land animals.
Kangaroos can hop.
Water is made up of hydrogen and oxygen.
Bats are nocturnal.
A rainbow has multiple colors.
Humans have teeth.
A bee can fly.
Water freezes at 0 degrees Celsius.
The Earth orbits the Sun.
Humans have 10 fingers.
Birds lay eggs.
The Pacific Ocean is the largest ocean.
Plants convert sunlight to energy.
Lead is heavier than aluminum.
The Eiffel Tower is in Paris.
Diamonds are made of carbon.
Bees make honey.
Mars is a planet in our solar system.
Dogs are mammals.
A year has 365 days.
The human body contains blood.
Mount Everest is the tallest mountain on Earth.
Fish live in water.
The piano is a musical instrument.
Apples grow on trees.
The heart pumps blood.
Tokyo is the capital of Japan.
Cats have whiskers.
Wood can burn.
Humans need oxygen to survive.
The Moon orbits the Earth.
Penguins cannot fly.
Salt is used to season food.
Bicycles have two wheels.
Rome is in Italy.
The sun rises in the east.
Books contain pages.
Tigers are carnivores.
Glass is made from sand.
Clouds are made of water vapor.
The Earth has one moon.
Lions live in Africa.
Carrots are vegetables.
Fire is hot.
Rain is wet.
Grass is green.
Elephants have trunks.
Snow is cold.
Bananas have peels.
Pencils contain graphite.
Eggs have shells.
Milk comes from cows.
Humans have two eyes.
Ice floats on water.
The sky appears blue.
Donald Trump is the current U.S. President.
Sunflowers face the sun.""".split("\n")


# In[3]:


false_eval = """Clouds are made of cotton candy.
Snakes have legs.
Leaves turn purple in autumn.
Bicycles run on electricity.
Stars are visible only in summer.
Mountains are formed overnight.
The Amazon River flows through Russia.
Tomatoes are blue when ripe.
Dolphins are a type of shark.
The Great Wall of China is in Australia.
A refrigerator makes food warmer.
Sharks are mammals.
Deserts are covered in snow.
Jupiter is the smallest planet.
Spiders have six legs.
Salt tastes sweet.
Babies can walk at birth.
Ants are larger than mice.
The heart pumps air through the body.
Lightbulbs produce darkness.""".split("\n")

true_eval = """Water boils at 100°C at sea level.
The heart is located in the chest.
Dolphins are mammals, not fish.
Spiders have eight legs.
Sound travels in waves.
Humans have 206 bones in their body.
Gold is a precious metal.
The speed of light is faster than sound.
The Earth's atmosphere contains nitrogen.
Whales breathe air through blowholes.
Antarctica is the coldest continent.
The Statue of Liberty is in New York.
Mercury is the closest planet to the Sun.
Computers use electricity to operate.
Honey is produced by bees.
The Great Wall of China is visible from Earth.
The human body contains approximately 60% water.
Most plants grow from seeds.
Humans have five senses.
Iron is attracted to magnets.""".split("\n")


# In[4]:


MAXLEN = 50


# In[5]:


print(len(false), len(true))
# don't ask why I need this sigh
assert len(false) == len(true)


# In[16]:


get_ipython().system('pip install torch')


# In[14]:


import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# device = "cuda"
device = "cpu"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
model = HookedTransformer.from_pretrained(model_name).eval().to(device)


# In[13]:


tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[5]:


def format(A, B):
  return f"""Human:
Is either of A or B true?
A={A}
B={B}
Assistant:"""


# In[6]:


ex = format(false[0],true[0])


# In[7]:


lyes = model.forward("number of legs on a dog: 4", return_type="loss").item()
lno = model.forward("number of legs on a dog: 3", return_type="loss").item()

print(lyes, lno)


# In[ ]:


# Collect FT data
FT_data = []
for A, B in zip(false, true):
    tokens = model.to_tokens(format(A, B))
    # Pad if needed
    if tokens.shape[1] < MAXLEN:
        padding = torch.full((1, MAXLEN - tokens.shape[1]), 0, device=tokens.device)
        tokens = torch.cat([tokens, padding], dim=1)
    _, cache = model.run_with_cache(tokens)
    FT_data.append(cache["blocks.7.hook_resid_post"])

# Collect TT data
TT_data = []
for A, B in zip(true, true):
    tokens = model.to_tokens(format(A, B))
    # Pad if needed
    if tokens.shape[1] < MAXLEN:
        padding = torch.full((1, MAXLEN - tokens.shape[1]), 0, device=tokens.device)
        tokens = torch.cat([tokens, padding], dim=1)
    _, cache = model.run_with_cache(tokens)
    TT_data.append(cache["blocks.7.hook_resid_post"])


# In[19]:


X = torch.cat(FT_data + TT_data)
Y = torch.cat([torch.zeros(len(FT_data)), torch.ones(len(TT_data))]).to(device)


# In[20]:


X1 = X[100:].reshape(100, -1)
X2 = X[:100].reshape(100, -1)


# In[21]:


X = X.cpu().numpy().reshape(200, -1)
y = Y.cpu().numpy()


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

LRmodel = LogisticRegression(random_state=42, max_iter=1000)
LRmodel.fit(X, y)

y_pred = LRmodel.predict(X)
y_prob = LRmodel.predict_proba(X)[:, 1]
accuracy = accuracy_score(y, y_pred)
print(accuracy)


# In[ ]:





# # junk below

# # JUNK

# In[ ]:


def pytorch_pca(X, n_components=2, COL='r'):
    n_samples, n_features = X.shape
    X_centered = X - torch.mean(X, dim=0)

    if n_features > n_samples:
        cov_matrix = torch.mm(X_centered, X_centered.t()) / (n_samples - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        eigenvectors = eigenvectors[:, :n_components]

        principal_components = torch.mm(X_centered.t(), eigenvectors)

        for i in range(n_components):
            principal_components[:, i] = principal_components[:, i] / torch.norm(principal_components[:, i])

        X_reduced = torch.mm(X_centered, principal_components)
    else:
        U, S, V = torch.svd(X_centered)

        X_reduced = torch.mm(X_centered, V[:, :n_components])

    # total_var = torch.sum(torch.var(X_centered, dim=0))
    # explained_var = torch.var(X_reduced, dim=0)
    # explained_var_ratio = explained_var / total_var

    X_reduced_np = X_reduced.cpu().numpy()
    # explained_var_ratio_np = explained_var_ratio.cpu().numpy()
    plt.scatter(X_reduced_np[:, 0], X_reduced_np[:, 1], c=COL, alpha=0.7, s=50)
    # return X_reduced, explained_var_ratio


# In[ ]:


# plt.figure(figsize=(10, 8))
# plt.grid(alpha=0.3)
# plt.tight_layout()
# pytorch_pca(X1,COL='r')
# pytorch_pca(X2,COL='b')
# plt.show()


# In[ ]:


# # question:
# # can we distinguish red vs blue by projecting onto the average direction?

# avg1 = torch.mean(X1, dim=0)
# avg2 = torch.mean(X2, dim=0)

# results = [X1 @ avg1, X1 @ avg2, X2 @ avg1, X2 @ avg2]
# colors = ['r','b','g','y']; ct = 0
# for res in results[2:]:
#     plt.hist(res.cpu().numpy(), bins=20, alpha=0.5, color=colors[ct])
#     ct += 1
# plt.show()

# # answer: nope definitely not!

