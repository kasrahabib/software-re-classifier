# kasrahabib/KM45L6V2TCTO

This model is a fine-tuned version of [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), for classifying software requirements into functional (F) and Non-functional (NF) types, on non-disclosed subset of Software Requirements Dataset (SWARD). It achieves the following results on the evaluation set; publicly disclosed subset of SWARD:
- Train Loss: 0.0458
- Validation Loss: 0.1759
- Epoch: 7
- Final Macro F1-score: 0.95

<b>Labels</b>: 
0 or F -> Functional;
1 or NF -> Non-functional;<br>

For a **quick demonstration** of the model's capabilities on Hugging Face, please click [here](https://huggingface.co/kasrahabib/KM45L6V2TCTO?text=The+email+string+consists+of+x%40x.x+and+is+less+than+31+characters+in+length+and+is+not+empty.).

## Usage Pipeline
```python
from transformers import pipeline

frame_work = 'tf'
task = 'text-classification'
model_ckpt = 'kasrahabib/KM45L6V2TCTO'

software_requirment_cls = pipeline(task = task, model = model_ckpt, framework = frame_work)

example_1_f = 'The START NEW PROJECT function shall allow the user to create a new project.'
example_2_nf = 'The email string consists of x@x.x and is less than 31 characters in length and is not empty.'
software_requirment_cls([example_1_f, example_2_nf])

```
```
[{'label': 'F', 'score': 0.9998922348022461},
 {'label': 'NF', 'score': 0.999846339225769}]
```

## Model Inference:
```python

import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_ckpt = 'kasrahabib/KM45L6V2TCTO'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt)

example_1_f = 'The START NEW PROJECT function shall allow the user to create a new project.'
example_2_nf = 'The email string consists of x@x.x and is less than 31 characters in length and is not empty.'
requirements = [example_1_f, example_2_nf]

encoded_requirements = tokenizer(requirements, return_tensors = 'np', padding = 'longest')

y_pred = model(encoded_requirements).logits
classifications = np.argmax(y_pred, axis = 1)

classifications = [model.config.id2label[output] for output in classifications]
print(classifications)
```
```
['F', 'NF']
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

model_ckpt = 'rest_of_the_path/KM45L6V2TCTO'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt)

example_1_f = 'The START NEW PROJECT function shall allow the user to create a new project.'
example_2_nf = 'The email string consists of x@x.x and is less than 31 characters in length and is not empty.'
requirements = [example_1_f, example_2_nf]

encoded_requirements = tokenizer(requirements, return_tensors = 'np', padding = 'longest')

y_pred = model(encoded_requirements).logits
classifications = np.argmax(y_pred, axis = 1)

classifications = [model.config.id2label[output] for output in classifications]
print(classifications)
```
```
[{'label': 'F', 'score': 0.9998922348022461},
 {'label': 'NF', 'score': 0.999846339225769}]
### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'Adam', 'weight_decay': None, 'clipnorm': None, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'jit_compile': True, 'is_legacy_optimizer': False, 'learning_rate': {'class_name': 'PolynomialDecay', 'config': {'initial_learning_rate': 2e-05, 'decay_steps': 1392, 'end_learning_rate': 0.0, 'power': 1.0, 'cycle': False, 'name': None}}, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'amsgrad': False}
- training_precision: float32


### Framework versions

- Transformers 4.26.1
- TensorFlow 2.11.0
- Datasets 2.10.1
- Tokenizers 0.13.2
