# A Punctuation-Robust Approach to English-to-Marathi Machine Translation

This repository contains the code, models, and resources for the final project submission for **CS 772 – FINAL PROJECT EVALUATION**.

The project addresses a critical issue in Machine Translation (MT): building systems that are robust to punctuation errors or omissions in the source English text, specifically for translation into the Marathi language.

---

## 🌟 Project Details

| Category | Details |
| :--- | :--- |
| **Title** | Punctuation-Robust Machine Translation: A Case Study of MT from English to Marathi |
| **Course** | CS 772 – FINAL PROJECT EVALUATION |
| **Submission Date** | 27th November 2025 |

### 👨‍💻 Authors and Guidance

| Role | Name | Student ID |
| :--- | :--- | :--- |
| **Author 1** | Kaustubh S. Shejole | 24M2109 |
| **Author 2** | Shalaka Thorat | 24M0848 |
| **Project Guide** | Sourabh Deoghare | N/A |

---

## 💡 Problem Statement

The goal is to create a robust English-to-Marathi Machine Translation system capable of handling English sentences provided **with or without punctuations** while preserving the intended meaning.

* **Input:** An English sentence (Latin script), potentially ambiguous due to missing punctuation (e.g., "Let's eat Grandma").
* **Output:** The corresponding translation in Marathi (Devanagari script), which accurately reflects the intended meaning (e.g., "Let's eat, Grandma").

### Motivation

Punctuation is a vital "signaling system" in language, primarily serving a **Grammatical Role** (marking segment boundaries, which can change meaning) and a **Rhetorical Role** (adding emphasis). Missing punctuation, especially the comma, often leads to meaning-changing ambiguities, which traditional MT models struggle to resolve.

---

## 📐 Methodology: Two Approaches

We explored two distinct methodological approaches to address punctuation-induced ambiguity.

### Approach 1: Pipeline Approach (Punctuation Restoration + Original MT)

This sequential approach uses the **IndicTrans2 en-indic** model as the base MT system.

1.  **Step 1: Punctuation Restoration:** The unpunctuated English input is first processed to restore correct punctuation.
    * **Models Explored:** Fine-tuned `bert-large-uncased`, `microsoft mpnet base` (Token-Classification Paradigm), and fine-tuned `google t5 base` / **AI4Bharat’s Cadence Model** (Text-to-Text Generation Paradigm).
2.  **Step 2: Translation:** The punctuation-restored English sentence is then translated by the original IndicTrans2 MT model.

> **Best Performance in Approach 1:** **Google T5-Base (fine-tuned) + Original IndicTrans2 Model.**

### Approach 2: Direct Fine-tuning of MT Models (IndicTrans2)

This approach involves directly fine-tuning the base **IndicTrans2** model on punctuation-robust datasets to implicitly learn the context.

* The **IITB-ENG-MAR** dataset was utilized to create four variants for fine-tuning:
    1.  Original data.
    2.  Data with all English source punctuations removed.
    3.  Combined Model (Alternate keeping/removing punctuation).
    4.  **Combined Model (Keeping and removing punctuation for each sentence, resulting in 2x data size).**

> **Best Performance in Approach 2:** **Combined Model (2x data size).**

---

## 🧪 Evaluation and Results

### Punctuation-English-Marathi (PEM) Test Benchmark

A significant contribution of this project is the creation of a novel, human-labeled test set named **Punct-Eng-Mar (PEM)**.

* **Size:** 54 instances.
* **Curation:** Ambiguous examples were manually curated based on punctuation literature (specifically, **John Kirkman's book on Punctuations**) where the absence of punctuation creates natural human ambiguity.
* **Focus:** The majority of instances (38 out of 54) focused on ambiguity caused by a missing comma.

### Key Findings (Qualitative)

Through qualitative analysis, the best models from both approaches successfully resolved complex ambiguous cases (e.g., distinguishing between "Honey" as a person's name versus the substance) by inserting the appropriate punctuation.

* The **T5 + Original (Approach 1)** model demonstrated the overall best performance, achieving "all correct" status in the qualitative case studies.

---


## 💻 Models

This project utilized various models, ranging from the baseline IndicTrans2 to several fine-tuned variants.

### Machine Translation Model Choices

| Model Name | Hugging Face Model ID |
| :--- | :--- |
| **Original IndicTrans2 (en-indic 200M)** | `ai4bharat/indictrans2-en-indic-dist-200M` |
| **Finetuned (Punctuation)** | `thenlpresearcher/iitb-en-indic-only-punct` |
| **Finetuned (No Punctuation)** | `thenlpresearcher/iitb-en-indic-without-punct` |
| **Combined Finetuned (1x Punct)** | `thenlpresearcher/shalaka_fd_indictrans2-en-indic-dist-200M_finetuned_eng_Latn_to_mar_Deva` |
| **Combined Finetuned (2x Punct)** | `thenlpresearcher/shalaka_indictrans2-en-indic-dist-200M_finetuned_eng_Latn_to_mar_Deva` |
| **t5-augmented Original IndicTrans2 (en-indic 200M)** | `ai4bharat/indictrans2-en-indic-dist-200M` |

### T5 Punctuation Restoration Model (Example Usage)

The fine-tuned T5 model used for Punctuation Restoration in **Approach 1** can be accessed via the Hugging Face pipeline:

```python
from transformers import pipeline

# Initialize the pipeline for text-to-text generation
punctuator_pipeline = pipeline("text2text-generation", model="thenlpresearcher/iitb-t5-finetuned-punctuation")

text = "the morning sky stretched over the city like a quiet sheet of pale blue while people hurried through the streets"

# Run the text through the pipeline
output = punctuator_pipeline(text, max_length=128)

# Sample Output: [{'generated_text': 'the morning sky stretched over the city like a quiet sheet of pale blue while people hurried through the streets.'}]

```

## 🛠️ Environment Setup and Dependencies

To ensure a smooth and reproducible environment for running the project code, two separate environments are recommended based on the required dependencies.

### General Project Setup

For running the core Machine Translation models and general project scripts, please utilize the following dependency file:

| Purpose | Requirement File | Installation Command |
| :--- | :--- | :--- |
| **General Dependencies** | `requirements.txt` | `pip install -r requirements.txt` |

### BLEURT Evaluation Environment

To perform evaluation using the **BLEURT** metric, a separate, specific environment is required to manage its dependencies effectively:

| Purpose | Requirement File | Installation Command |
| :--- | :--- | :--- |
| **BLEURT Metric Dependencies** | `bleurt_env_requirements.txt` | `pip install -r bleurt_env_requirements.txt` |

It is highly recommended to use **virtual environments** (e.g., `venv` or `conda`) to isolate these dependencies and prevent conflicts with other system packages.

---



## 🔗 Project Files

* **Project Presentation (PPT):** [CS 772 – FINAL PROJECT EVALUATION](https://docs.google.com/presentation/d/1Wn99JgCa23sDjOis7VcCl_dJdGSxpebdkykqVnT05xQ/edit?usp=sharing)

---


## 🌐 Demo

A live demonstration of the project, showcasing the best-performing models, has been hosted on Huggingface Spaces.

**[Demo Link](https://huggingface.co/spaces/thenlpresearcher/Punctuation_Robust_English_to_Marathi_Translation_Kaustubh_Shalaka)**

---

## 🚀 Future Work

We have identified several promising avenues for future research:

1.  **Metric Development:** Developing a robust metric to specifically evaluate whether the **intended meaning is preserved** (as this is currently checked manually).
2.  **Advanced Approaches:** Investigating more intelligent, unified, or end-to-end approaches to address the ambiguity problem.
3.  **Language Extension:** Expanding the study to include other Indic languages to analyze the effects of punctuation across a broader linguistic context.
4.  **Error and Bias Handling:** Addressing identified issues, such as gender-bias in certain fine-tuned models, and improving the models' ability to correct intentionally wrong input punctuations.

---

## 🙏 Acknowledgments

We would like to express our gratitude to the following individuals and groups:

* **Sourabh Deoghare:** For his constant guidance and support throughout the project.
* **Course Instructors:** For offering CS 772, which provided the foundational knowledge necessary for this research.
* **Evaluators of the Course Project:** Your suggestions and feedback are invaluable and will help us refine our work.