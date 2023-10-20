# Paper: Uncovering the Causes of Emotions in Software Developer Communication Using Zero-shot LLMs
In the following, we briefly describe the components included in this project and the software required to run the experiments.

## Project Structure
The project includes the following files and folders:

  - __/annotation__: A folder that contains annotation instructions and the annotated file.
 - __/blue_score__: A folder that contains the scripts for BLEU score computation for emotion-cause extraction.
 - __/cluster__: A folder that contains the scripts for generating clusters for the case study.
 - __/Emotion_Classification__: A folder that contains scripts for training and testing emotion classification using BERT and RoBERTa.
      - emotion_classification.py: the script of emotion classification using fine-tuned LLMs.
      - esem-e.py: the script to run eseme mode.
      - ReadMe.md: contains the readMe on how to run emotion_classification.py and esem-e.py.
      - __/Dataset__: A folder containing the experiment's datasets.
- __/Prompts__: A folder that contains prompts that have been used in this experiment.
