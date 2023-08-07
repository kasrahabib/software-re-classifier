# Deep Learning-based Classification of Software Requirements: A Large-Scale Dataset and Evaluation

## Short Description:
This project is the efforts of a research endeavor aimed at establishing the initial extensive Software Requirements Dataset (SWARD), in requirements engineering domain (RE), and employing more generalisable transfer learning to meticulously fine-tune universally adaptable models. These models are intended for the classification of software requirements into functional and non-functional categories, encompassing 19 distinct types of non-functional subclass requirements.


| Paper Abstract  |
|-----------------------------------------|
|**Purpose**: Automatic classification software requirments using machine and deep learning has so far lacked generalization due to the absence of a large-scale and high-quality labeled requirements dataset. In this study, we introduce a large-scale software requirements dataset, then investigate the generalization capabilities of the transfer learning-based approach by fine-tuning classifiers on varying subsets of this dataset.<br>
**Methods:** We created the Software Requirements Dataset (SWARD) by combining publicly disclosed and non-disclosed software projects. Next, we apply a novel approach to labeling requirements as functional (F) and non-functional (NF), including 19 NF subclasses. Finally, we create different subsets of the SWARD and use transfer learning to classify requirements. We evaluate the trained models and compare our results with previous works.<br>
**Results** SWARD includes 43,777 requirement texts, with ca. 76% (33,098) from publicly disclosed projects, that will be published in a standardized format with extensive meta-information. Our proposed models, achieving a macro f1-score of 99% for classifying F and NF requirements and 79% for NF subclasses, will be made publicly available. These results outperform the state-of-the-art model, NoRBERT, with a 13% f1-score increase for F and NF classification and with a >40% increase for NF subclasses classification including nine new subclasses.<br>
**Conclusion:** Using SWARD, we were able to achieve measurable improvements in classification tasks for software requirements, overcoming the challenges to the generalization capability of machine and deep leanring-based approaches that commonly occur with smaller datasets.|

## Project Content:
* [f_nf_classifier_model](https://github.com/kasrahabib/software-re-classifier/tree/main/f_nf_classifier_model) contains three fine-tuned models for the binary classification of software requirements into functional and non-functional requirements.
* [nf_subclasses_classifier_model](https://github.com/kasrahabib/software-re-classifier/tree/main/nf_subclasses_classifier_model/KM35NCDF) contains one fine-tuned model for the classification of 19 non-functional software requirements.<br>
**Note:** For usage and more details, navigate to the subdirectories.

## Acknowledgment:
We want to express our appreciation to the Hugging Face API for its pivotal role in our research endeavor. Our models are built upon the foundation of transfer learning, drawing from the reservoir of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model hosted on Hugging Face. The project owes a debt of gratitude to the open-source principles fostered by Hugging Face, which have undeniably contributed to shaping the triumph of our undertaking. It is with great pride that we align ourselves with this trailblazing community of innovators.



