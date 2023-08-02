# Emotion Classification Script

This script performs emotion classification using various transformer-based models such as BERT and RoBERTa. It uses PyTorch and the Hugging Face Transformers library to build and train the classification model. The script supports both traditional and contrastive fine-tuning approaches for the models.

## Requirements

To run this script, you need the following dependencies:

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- pandas
- nltk

Install the required packages using the following command:

`pip install torch transformers pandas nltk`



## Usage

1. Make sure you have training and test data in CSV format. The CSV files should contain two columns: "text" (containing the text to classify) and "label" (containing the class labels). The label value should be Anger, Fear, Love, Joy, Surprise, Sadness. 


2. Run the emotion classification script with the following command:


`python emotion_classification.py --epoch EPOCH --delta DELTA --batch_size BATCH_SIZE --col COLUMN --model_name MODEL_NAME --output OUTPUT_FILE --train_file TRAIN_CSV --test_file TEST_CSV`


### Arguments:

- `EPOCH`: Number of epochs for training (default: 100).
- `DELTA`: Early stopping criterion. Training will stop if the average training loss goes below this value (default: 0.01).
- `BATCH_SIZE`: Batch size for training (default: 128).
- `COLUMN`: The emotion column to classify (choose from Anger, Fear, Love, Joy, Surprise, Sadness) (required).
- `MODEL_NAME`: Model architecture to use (choose from bert, roberta) (required).
- `OUTPUT_FILE`: Output file name for storing predictions (default: output.csv).
- `TRAIN_CSV`: Path to the training CSV file (required).
- `TEST_CSV`: Path to the test CSV file (required).

## Output

The script will print the training progress, average training loss, average validation loss, and validation F1-score for each epoch. Additionally, it will save the predictions in the specified output file in CSV format, containing columns "Pred" (predicted labels) and "True" (true labels).

## Note

- Ensure you have the appropriate training data for emotion classification in the CSV format, with one column containing the text data and another column containing the corresponding emotion labels.

- Ensure that your system has access to a CUDA-enabled GPU if you want to use GPU acceleration for training. The script automatically checks for GPU availability and uses it if available.
