#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
"""
The code for creating the dataset from T-REx for pre-training factual adapter on relation classification task.

Input：cleaned T_Rex dataset. "T_REx_example.json"
Output：examples: {tokens, obj_label(relation label), entity [subj_strat,subj_end]}

- Whole dataset   "T_REx_examples_{}_{}.json".format(max_count,min_predicate_num)
- Training split   data/train.json
- Dev split    data/dev.json
- labels    data/relations.json

example：
{'docid': 'Q3343462',
 'token': ['Nora','Zaïdi','(','Nora','Mebrak','-','Zaïdi','),','born','on','July','6',',','1965','in','Bethoncourt','(','French','département','of',
  'Doubs','),','daughter','of','an','Algerian','textile','toiler',',','is','a','French','activist','who','seated','in','the','European','Parliament','from',
  '1989','to','1994','.'],
  'relation': 'P31',
  'subj_start': 20,
  'subj_end': 20,
  'obj_start': 18,
  'obj_end': 18,
  'subj_label': 'Q3361',
  'obj_label': 'Q6465'}
"""

import json
# import matplotlib.pyplot as plt
import os

# In[2]:


def get_property2idx_dict(file="../data/cleaned_T_REx/T_REx_pred2ic.json"):
    with open(file, "r") as fin:
        property2idx_dict = json.load(fin)
    return property2idx_dict
# property dictionary.
property2idx_dict = get_property2idx_dict()
idx2property_dict = {v: k for k, v in property2idx_dict.items()}

print(len(property2idx_dict.keys()))

# In[ ]:


def load_examples(file):
    with open(file, "r") as fin:
        all_examples = json.load(fin)
    print("total examples：", len(all_examples))
    return all_examples
examples = load_examples(file="../data/cleaned_T_REx/T_REx_example.json")


# In[ ]:


example = examples[0]
print(example)
subj_start_id = example["subj_start"]
subj_end_id = example["subj_end"]
example["token"][subj_start_id:subj_end_id+1]

# In[ ]:


sub_examples = examples
examples_fpre = {}
for i, example in enumerate(sub_examples):
    if example['relation'] in examples_fpre.keys():
        temp = examples_fpre[example['relation']]
        temp.append(example)
        examples_fpre[example['relation']]= temp
    else:
        examples_fpre[example['relation']]= [example]


# In[ ]:


from collections import Counter
count_list = []
label_list = []
total_predicates_uri = {}
for key in examples_fpre.keys():
    total_predicates_uri[key] = len(examples_fpre[key])

# In[ ]:


print(total_predicates_uri['P17'])
print(total_predicates_uri['P131'])
print(total_predicates_uri['P31'])
print(total_predicates_uri['P47'])

# In[ ]:


# counting
counter_predicates_uri = Counter(total_predicates_uri)
sort_counter_predicates_uri = counter_predicates_uri.most_common(200)
labels, values = zip(*sort_counter_predicates_uri)

my_labels = list(labels)
my_labels.reverse()
my_values = list(values)
my_values.reverse()

fig, ax = plt.subplots(figsize=(10, 30),dpi=100)
b = ax.barh(range(len(my_labels)), my_values,height=0.8, color='steelblue',alpha=0.7) 
for rect in b:   
    w = rect.get_width()    
    ax.text(w, rect.get_y()+rect.get_height()/2, '%d' % 
            int(w), ha='left', va='center')

ax.set_yticks(range(len(my_labels)))
ax.set_yticklabels(my_labels)

plt.xticks(()) 
plt.title('Total Subset Predicates distribution', loc='center')
plt.savefig('total_subset_predicates_dis.png')
plt.show()



# In[ ]:


# del P31 (instance of)
del examples_fpre['P31']

# In[ ]:


sub_examples = examples_fpre

# In[ ]:


# del some relations
total_examples = 0
min_predicate_num = 50
count = 0
discard_list = []
for key in sub_examples.keys():
    if len(sub_examples[key])<min_predicate_num:
        discard_list.append(key)
        count+=1
    else:
        total_examples+=len(sub_examples[key])
for key in discard_list:
    del sub_examples[key]
    
print('the number of relations which have examples less than {} is {} '.format(min_predicate_num,count))
print('the remaining realtions:',len(sub_examples.keys())-count)
print("total examples：", total_examples)

# In[ ]:


counter_predicates_uri.most_common()

# In[ ]:


# counting
from collections import Counter

count_list = []
label_list = []
total_predicates_uri = {}
for key in sub_examples.keys():
    total_predicates_uri[key] = len(sub_examples[key])

counter_predicates_uri = Counter(total_predicates_uri)
sort_counter_predicates_uri = counter_predicates_uri.most_common(200)
# labels, values = zip(*counter_relations.items())
labels, values = zip(*sort_counter_predicates_uri)

my_labels = list(labels)
my_labels.reverse()
my_values = list(values)
my_values.reverse()

fig, ax = plt.subplots(figsize=(10, 30),dpi=100)
b = ax.barh(range(len(my_labels)), my_values,height=0.8, color='steelblue',alpha=0.7)
for rect in b:   
    w = rect.get_width()    
    ax.text(w, rect.get_y()+rect.get_height()/2, '%d' % 
            int(w), ha='left', va='center')

ax.set_yticks(range(len(my_labels)))
ax.set_yticklabels(my_labels)

plt.xticks(()) 
plt.title('Subset Predicates distribution', loc='center')
plt.savefig('subset_predicates_dis.png')
plt.show()


# In[ ]:


examples = []
for key in sub_examples.keys():
    examples.extend(sub_examples[key])
print(len(examples))
print(len(sub_examples.keys()))

# In[ ]:


pred2id = {}
for example in examples:
    if example["relation"] in pred2id:
        continue
    else:
        pred2id[example["relation"]] = len(pred2id)

# In[ ]:


import random
random.shuffle(examples)
train_examples = examples[:int(len(examples) * 0.9)]
val_examples = examples[int(len(examples)*0.9):]

# In[ ]:


# saving
save_path = "../data/trex-rc"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
fout_pred2id = open(os.path.join(save_path,"relations.json"), "w")
fout_train = open(os.path.join(save_path,"train.json"), "w")
fout_val = open(os.path.join(save_path,"dev.json"), "w")

json.dump(pred2id, fout_pred2id)
json.dump(train_examples, fout_train)
json.dump(val_examples, fout_val)

fout_pred2id.close()
fout_train.close()
fout_val.close()

# In[ ]:


relations = load_examples(file="../data/trex-rc/relations.json")

# In[ ]:


relations.keys()
