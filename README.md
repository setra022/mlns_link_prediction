# MLNS : Link Prediction

Run the following to create a json file that stores text features :
```bash
python generate_text_features
```

In `main.py` :
- set `validate_model` to True to evaluate the model, and False to create test set predictions.
- `data_path`: to load/save features in a pickle file