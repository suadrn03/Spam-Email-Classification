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

Open [`spam_classifier_fixed.ipynb`](spam_classifier_fixed.ipynb) in Jupyter Notebook, JupyterLab, or GitHub's notebook viewer. The notebook includes section headings, explanations of each task, and comments to make the workflow easier to follow.

## Dataset

The notebook expects a CSV file named `spam_ham_dataset.csv` in the working directory. Based on the code, the file must contain:

- `text`: the email message text.
- `label_num`: the numeric class label (the code assumes `1` means spam and `0` means ham).

The dataset is intended for public use. No separate license file is included for the dataset in this repository. Public availability alone does not establish redistribution rights, so users should check the original dataset source and any applicable terms before reusing or redistributing it.

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
4. Open `spam_classifier_fixed.ipynb`.
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

The corrected notebook fixes the original confusion-matrix helper, calculates logistic-regression validation loss on the validation split, restores the best weights after early stopping, and adds numerical safeguards. Model selection uses validation data, with the test set reserved for final evaluation. Run the notebook with the actual dataset to generate and verify your own results.

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
- Document the original dataset source and any usage terms.
- Save evaluation plots and include a small results table after running the notebook on the actual dataset.

## License and dataset use

No separate license has been added to this repository. The dataset is described as intended for public use, but this statement is not a substitute for permission or terms from its original publisher. Check the dataset source before redistribution. The repository code also has no explicit open-source license, so standard copyright rules apply unless one is added.
