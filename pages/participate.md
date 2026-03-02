# How to participate

## What to submit
Submit a single Python file named `submission.py`.

It must define a function:

```python
def get_model():
    """
    Return an *untrained* scikit-learn compatible model.
    The ingestion program will call:
      - model.fit(X_train, y_train)
      - model.predict(X_test)
    """
    ...