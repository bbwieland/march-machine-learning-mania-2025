# Kaggle March Madness 2025 Submission

## Instructions

The script for prediction can be run from the command like:

```
python run_model.py
```

You must add the flags `league` and `season` for the script to execute:

- `league` is either "M" or "W" and specifies whether predictions should be made for NCAAM or NCAAW teams.

- `season` specifies the season for which team strengths and match results should be modeled.

For example, a valid command to make predictions for this season's NCAAW bracket would be:

```
python run_model.py --season=2025 --league="W"
```