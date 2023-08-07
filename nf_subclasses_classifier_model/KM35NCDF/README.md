---
license: apache-2.0
tags:
- generated_from_keras_callback
model-index:
- name: kasrahabib/KM35NCDF
  results: []
widget:
- text: "Application needs to keep track of subtasks in a task."
  example_title: "Requirment 1"
- text: "The system shall allow users to enter time in several different formats."
  example_title: "Requirment 2"
- text: "The system shall allow users who hold any of the ORES/ORELSE/PROVIDER keys to be viewed as a clinical user and has full access privileges to all problem list options."
  example_title: "Requirment 3"
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# kasrahabib/KM35NCDF

This model is a fine-tuned version of [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on Software Requirements Dataset (SWARD) for classifying 19 Non-functional requirements. Note that based on literature, two out of 19 classes are Data and Behavior, which belong to types of Functional software requirements. It achieves the following results on the evaluation set:
- Train Loss: 0.1691
- Validation Loss: 0.7548
- Epoch: 14
- Final Macro F1-score: 0.79


<b>Labels</b>: 
0 or A -> Availability;
1 or AC -> Access Control;
2 or AU -> Audit;
3 or B -> Behaviour;
4 or D -> Data;
5 or FT -> Fault Tolerance;
6 or I -> Interface/Interoperability;
7 or LE -> Legal;
8 or LF -> Look and Feel;
9 or MN -> Maintainability;
10 or O -> Operational;
11 or PE -> Performance;
12 or PO -> Portability;
13 or RL -> Reliability;
14 or SA -> Safety;
15 or SC -> Scalability;
16 or SE -> Security;
17 or ST -> Stability;
18 or US -> Usability;


## Usage Pipeline
```python
from transformers import pipeline

frame_work = 'tf'
task = 'text-classification'
model_ckpt = 'kasrahabib/KM35NCDF '

software_requirment_cls = pipeline(task = task, model = model_ckpt, framework = frame_work)

example_1_US = 'Application needs to keep track of subtasks in a task.'
example_2_PE = 'The system shall allow users to enter time in several different formats.'
example_3_AC = 'The system shall allow users who hold any of the ORES/ORELSE/PROVIDER keys to be viewed as a clinical user and has full access privileges to all problem list options.'

software_requirment_cls([example_1_US, example_2_PE, example_3_AC])
```
```
[{'label': 'US', 'score': 0.9712953567504883},
 {'label': 'PE', 'score': 0.9457865953445435},
 {'label': 'AC', 'score': 0.9639136791229248}]
```

## Model Inference:
```python

import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_ckpt = 'kasrahabib/KM35NCDF '
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt)

example_1_US = 'Application needs to keep track of subtasks in a task.'
example_2_PE = 'The system shall allow users to enter time in several different formats.'
example_3_AC = 'The system shall allow users who hold any of the ORES/ORELSE/PROVIDER keys to be viewed as a clinical user and has full access privileges to all problem list options.'
requirements = [example_1_US, example_2_PE, example_3_AC]

encoded_requirements = tokenizer(requirements, return_tensors = 'np', padding = 'longest')

y_pred = model(encoded_requirements).logits
classifications = np.argmax(y_pred, axis = 1)

classifications = [model.config.id2label[output] for output in classifications]
print(classifications)
```
```
['US', 'PE', 'AC']
```

## Usage Locally Downloaded (e.g., GitHub):


  1  - Clone the repository:
```shell
git lfs install
git clone url_of_repo
```
  2  - Locate the path to the downloaded directory <br>
  3  - Write the link to the path in the ```model_ckpt``` variable <br>
    
Then modify the code as below:
```python
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_ckpt =  'rest_of_the_path/KM35NCDF '
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt)

example_1_US = 'Application needs to keep track of subtasks in a task.'
example_2_PE = 'The system shall allow users to enter time in several different formats.'
example_3_AC = 'The system shall allow users who hold any of the ORES/ORELSE/PROVIDER keys to be viewed as a clinical user and has full access privileges to all problem list options.'
requirements = [example_1_US, example_2_PE, example_3_AC]

encoded_requirements = tokenizer(requirements, return_tensors = 'np', padding = 'longest')

y_pred = model(encoded_requirements).logits
classifications = np.argmax(y_pred, axis = 1)

classifications = [model.config.id2label[output] for output in classifications]
print(classifications)
```

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'Adam', 'weight_decay': None, 'clipnorm': None, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'jit_compile': True, 'is_legacy_optimizer': False, 'learning_rate': {'class_name': 'PolynomialDecay', 'config': {'initial_learning_rate': 2e-05, 'decay_steps': 6735, 'end_learning_rate': 0.0, 'power': 1.0, 'cycle': False, 'name': None}}, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'amsgrad': False}
- training_precision: float32

### Framework versions

- Transformers 4.26.1
- TensorFlow 2.11.0
- Datasets 2.10.0
- Tokenizers 0.13.2
