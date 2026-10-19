# Email Spam Classification with NumPy

A notebook-based machine-learning project exploring how to classify email messages as **spam** or **ham** (not spam). The notebook implements a linear classifier and a logistic-regression model using NumPy and SciPy sparse matrices, with text features created using a bag-of-words representation.

## Project goals

- Load and inspect a labeled email dataset.
- Split the data into training, validation, and test sets.
- Convert email text into numeric count features.
- Implement and compare two classification approaches.
- Tune learning rates, training duration, stopping tolerance, and the decision threshold.
- Evaluate predictions using accuracy, precision, recall, F1 score, and confusion matrices.
- Plot training/validation loss and accuracy.

## Notebook

Open [`spam_classifier.ipynb`](spam_classifier.ipynb) in Jupyter Notebook, JupyterLab, or GitHub's notebook viewer. The notebook includes section headings, explanations of each task, and comments to make the workflow easier to follow.

## Dataset

This project uses the [Spam Mails Dataset](https://www.kaggle.com/datasets/venkateshgarnepudi/spam-mails-dataset), published on Kaggle by Venkatesh Garnepudi. The uploader describes it as a CSV prepared from the **Enron1** spam/ham email collection. The Kaggle dataset listing specifies **CC0: Public Domain**; this is the uploader's stated license, and the original source's terms should also be considered if redistributing the emails.

Download `spam_ham_dataset.csv` from the linked dataset page and place it in the same folder as the notebook. The notebook uses:

- `text`: email message text.
- `label_num`: numeric class label (`1` for spam, `0` for ham).

The CSV is not required to be committed to this repository; downloading it separately keeps the repository lightweight.

## Requirements

The notebook uses:

- Python 3
- pandas
- NumPy
- Matplotlib
- scikit-learn
- SciPy
- Jupyter Notebook or JupyterLab

Install the main dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn scipy jupyter
```

## How to run

1. Clone or download this repository.
2. Place `spam_ham_dataset.csv` in the project directory.
3. Install the dependencies listed above.
4. Open `spam_classifier.ipynb`.
5. Run the cells from top to bottom.

## Workflow overview

1. **Data loading:** inspect the dataset and required columns.
2. **Data splitting:** create training, validation, and test partitions with stratification.
3. **Text vectorization:** use `CountVectorizer` to transform messages into bag-of-words features. The vocabulary is fit on training text only.
4. **Linear baseline:** train a linear model with gradient descent and tune its learning rate, number of epochs, and classification threshold.
5. **Logistic regression:** implement a sigmoid-based classifier and track training/validation metrics.
6. **Model selection:** choose settings with validation data.
7. **Final evaluation:** evaluate the chosen model on the held-out test set and visualize classification errors.

## Implementation notes

The models are implemented using NumPy and SciPy sparse matrices. Training and validation losses and accuracies are tracked during fitting. Hyperparameters and classification thresholds are selected using validation data, with the test set reserved for final evaluation. Run the notebook from top to bottom to reproduce the results.

## Metrics

- **Accuracy:** share of all messages classified correctly.
- **Precision:** share of messages predicted as spam that really are spam.
- **Recall:** share of spam messages correctly detected.
- **F1 score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts correct and incorrect predictions by class.

## Possible future improvements

- Add a reproducible environment file such as `requirements.txt`.
- Compare against a scikit-learn baseline such as `LogisticRegression` or `MultinomialNB`.
- Report per-class metrics and compare models using precision, recall, and F1 rather than accuracy alone.
- Save evaluation plots and include a small results table after running the notebook on the actual dataset.

## License

The dataset is listed as **CC0: Public Domain** on its [Kaggle page](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data). That listing does not automatically license the code in this repository. No separate license has been added for the project code; unless one is added, standard copyright rules apply.
