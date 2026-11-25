# NLP for Customer Review Analysis - Academic project

Classification and regression on the [Fake Review dataset](https://huggingface.co/datasets/debojit01/fake-review-dataset/tree/main) (40,400 reviews) to predict a review's rating (1–5) from its text.
More details on: https://portfolio.noknok.dev/projects/analyse_automatique_de_donnees

## Approach

1. **NLP preprocessing** (spaCy): remove stop-words/punctuation/HTML, keep nouns/verbs/adjectives/adverbs, lemmatize.
2. **Vectorization**: TF-IDF with bigrams (captures nuances like "not good").
3. **Feature encoding**: OneHotEncoder for category, ColumnTransformer to pipeline text + category + label.
4. **Split**: 80/20 train/test with under-sampling for balance.

## Models tested (scikit-learn)

| Task | Models | Best result |
|---|---|---|
| Binary classification (pos/neg) | KNN, Decision Tree, Random Forest, Logistic Regression | Logistic Regression — accuracy **0.65** |
| Multiclass classification (1–5) | same | Logistic Regression — accuracy **0.31** (unusable) |
| Regression (rating 1–5) | Linear, Ridge, Lasso | Ridge — RMSE **1.18** (vs 8.82 without NLP) |

Hyperparameters tuned with `GridSearchCV` + cross-validation.

## Key takeaway

Model performance depends more on data quality and preparation than on algorithm choice. The spaCy cleaning/balancing/filtering pipeline was essential.

## Installation & reproduction

```bash
# Clone and enter the project
git clone <repo-url>
cd <project-dir>

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install scikit-learn spacy pandas numpy

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Run the notebook / script
jupyter notebook notebooks/analysis.ipynb
# or
python src/main.py
```