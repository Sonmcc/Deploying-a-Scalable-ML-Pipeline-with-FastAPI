# Deploying a Scalable ML Pipeline with FastAPI

**GitHub Repository:** https://github.com/Sonmcc/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

## What is this project?

This project takes U.S. Census data and uses it to predict whether someone earns more or less than $50,000 a year. It walks through the full process of building a machine learning project the way it's often done in the real world:

1. Explore and understand the data
2. Build and train a machine learning model
3. Test that the code actually works (unit tests)
4. Check how the model performs across different groups of people (not just overall)
5. Wrap the model in a web API so other programs can send it data and get predictions back
6. Set up automatic checks (CI) so that every time code is pushed to GitHub, it's tested automatically

This was built as a course project for Udacity's "Deploying a Scalable ML Pipeline with FastAPI" course.

## What's in this repo?

- `data/census.csv` — the dataset used to train the model
- `ml/data.py` — code that cleans and prepares the data
- `ml/model.py` — code that trains the model, makes predictions, and checks its accuracy
- `train_model.py` — the script that actually runs the training process
- `slice_output.txt` — a report showing how well the model performs for different subgroups (like different education levels or countries)
- `model_card.md` — a plain-language summary of the model: what it does, how it was trained, how well it works, and its limitations
- `main.py` — the web API (built with FastAPI) that lets you send it data and get a prediction back
- `local_api.py` — a small script to test that the API is working correctly
- `test_ml.py` — automated tests that check the model code works as expected
- `screenshots/` — screenshots showing the tests and API working successfully

## How to run this project yourself

### 1. Set up your environment

You'll need Python 3.11. Create a virtual environment and install the requirements:

```bash
python3.11 -m venv fastapi
source fastapi/bin/activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This trains the model on the census data and saves it to the `model/` folder. It also creates `slice_output.txt`, showing performance broken down by group.

### 3. Run the tests

```bash
pytest -v
```

### 4. Check code style

```bash
flake8 .
```

### 5. Run the API

```bash
uvicorn main:app --reload
```

In a separate terminal, you can then test it with:

```bash
python local_api.py
```

## Continuous Integration

Every time code is pushed to this repo, GitHub Actions automatically runs the tests and style checks above, so problems get caught early. You can see the results under the "Actions" tab of this repository.