# Introduction

## Importance of Knowing What / When to Transfer

**Transfer learning** has emerged as a pivotal technique in machine learning and deep learning, enabling models to leverage knowledge from one task to enhance performance on related tasks. This approach is particularly valuable when labeled data for the target task is scarce, as it allows models to generalize more effectively by utilizing insights gained from previous learning experiences. 

For instance, when transferring models trained on large text corpora to a specialized domain such as medical or legal text, transfer learning can help improve performance when there is insufficient domain-specific labeled data. The general language model, such as GPT or BERT, transfers general linguistic knowledge to the target domain, aiding better text understanding in the target task {cite:p}`2019BERT`. In computer vision, models pretrained on general image datasets like ImageNet can be fine-tuned for specific tasks, such as medical image diagnosis (e.g., detecting tumors). This transfer is particularly effective when the target task lacks a large labeled dataset, as the pretrained model has already learned useful low- and mid-level features like edges and textures {cite:p}`2014ImageNet`. If negative consequences of transferability is not concerned in transfer learning, it can also lead to negative transfer if the source and target tasks are too dissimilar. Negative transfer occurs when transferring knowledge from a source task degrades the performance of the target task, often because the underlying task distributions are incompatible {cite:p}`5288526`.

## Informal Definition, Importance, and Criterias of Transferability

**Transferability** refers to the extent to which knowledge acquired from a source task can be applied to a target task. An informal definition of transferability is the ability of a model to apply learned representations or parameters from one sample, dataset or task to another, thereby improving learning efficiency and performance on the target sample, dataset or task. Transferability was first termed with empirical studies investigating how transferable knowledge is in neural networks {cite:p}`NIPS2014_375c7134`. Notably, the Taskonomy project established relationships between various visual tasks, providing a foundational understanding of task relationships and their impact on transferability {cite:p}`DBLP:journals/corr/abs-1804-08328`. Additionally, analytical metrics have been developed to quantify transferability, offering insights into the effectiveness of knowledge transfer between tasks. 

The significance of transferability spans multiple disciplines and applications. In battery research, for instance, models trained on data from one type of battery can be adapted to predict the performance of different battery chemistries, thereby accelerating the development of new technologies {cite:p}`10096896`. Similarly, in physics-based domains, transferring knowledge from simulations to real-world applications can lead to more accurate models and predictions. Moreover, transferability plays a vital role in various learning paradigms, such as model selection and continual learning, where the ability to apply knowledge from one task to another can significantly enhance learning efficiency and performance.

In order to better utilize the transferbability information to help with certain tasks, the criteria for good and effective transferability metrics is essential for assessing and facilitating knowledge transfer. For instance, in pre-hoc and post-hoc conditions transferability criteria differs. Pre-hoc metrics aim to estimate transferability before training, are computable, and offer theoretical interpretations or operational meanings. Post-hoc metrics, on the other hand, evaluate transferability after training has occurred. The diversity in the usage of transferbility as well as their corresponding criteria leads to varying definitions of transferability, making it challenging to compare and select metrics fairly.


## Existing Reviews

The concept of transferability in machine learning has been extensively explored, yet a unified theory remains elusive due to the diverse definitions and criteria proposed. A comprehensive survey by Pan and Yang examined various transfer learning approaches, highlighting the challenges in defining and measuring transferability {cite:p}`jiang2022transferabilitydeeplearningsurvey`. Jiang presented a survey connecting different areas in deep learning with their relation to transferability, aiming to provide a unified view {cite:p}`zhuang2020comprehensivesurveytransferlearning`. More recently, Xue systematically categorizes and qualitatively analyzes four types of prominent transferability estimation methods, offering insights for selecting appropriate methods {cite:p}`10639517`. These reviews underscore the lack of a singular, universally accepted definition of transferability. 


## Our Contributions

Therefore, the goal of this survey is to establish a unified definition and taxonomy of transferability metrics. We propose a taxonomy on transferability metrics based on the knowledge modality to transfer and the granularity at which transferability is evaluated. By systematically categorizing and analyzing existing metrics, we aim to provide a comprehensive understanding of transferability, thereby guiding future research and relating applications. Our contributions can be summarized as follows,

- **A Unified Framework**: We systematically review nine diverse learning paradigms along with transfer learning, and present a unified framework of transferability. 
- **A Comprehensive Survey on Transferablity**: This survey provides a thorough review of methodologies for various transfer learning problems, including out-of-distribution detection (OOD), domain adaptation (DA), and more. We connect these methodologies to offer insights into their interconnections and shared challenges.
- **Future Research Directions**: We conclude the survey with a discussion on open challenges and opportunities for future research, particularly in areas where transferability metrics can be further improved or extended.


```{bibliography} references.bib
```
