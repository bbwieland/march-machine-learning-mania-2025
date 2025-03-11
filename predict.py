import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from itertools import combinations
import constants as c
import utils

from typing import Dict, Any

def create_posterior_team_pairings(team_ids: np.array) -> Dict[str, Any]:
    """Given a set of team IDs, creates the posterior combination of all teams to predict over.

    Parameters
    ----------
    team_ids : np.array
        The team IDs to create pairings for.

    Returns
    -------
    Dict[str, Any]
        A data dictionary of each possible matchup to predict over.
    """
    team_pairings = list(combinations(team_ids, r=2))
    posterior_team_ids = [int(pair[0]) for pair in team_pairings]
    posterior_oppo_ids = [int(pair[1]) for pair in team_pairings]
    posterior_location = np.zeros(shape=len(team_pairings), dtype=int)

    posterior_team_ids = [np.where(team_ids == home_id)[0][0] for home_id in posterior_team_ids]
    posterior_oppo_ids = [np.where(team_ids == away_id)[0][0] for away_id in posterior_oppo_ids]

    posterior_data_dict = {
        "home_idx" : posterior_team_ids, 
        "away_idx" : posterior_oppo_ids,
        "at_home" : posterior_location 
    }

    return posterior_data_dict

def predict_posterior(model: pm.Model, trace: az.InferenceData, new_data: pm.Model) -> az.InferenceData:
    """Samples the supplied model & trace on new_data

    Returns
    -------
    az.InferenceData
        The posterior predictive trace
    """
    
    with model:
        pm.set_data(new_data=new_data)
        post_pred = pm.sample_posterior_predictive(trace=trace.sel(draw=range(250)))

    return post_pred

def get_game_predictions_from_posterior(post_pred: az.InferenceData, team_ids: np.array) -> pd.DataFrame:
    """Given the posterior predictive trace and the team ID encoder, extracts win probability predictions.

    Parameters
    ----------
    post_pred : az.InferenceData
        The posterior predictive trace from `predict_posterior`
    team_ids : np.array
        The numpy array containing team IDs, from our `pd.factorize` step earlier.

    Returns
    -------
    pd.DataFrame
        A DataFrame of game predictions.
    """

    win_probabilities = post_pred.posterior_predictive.home_win.mean(('chain', 'draw')).to_numpy()
    home_team = team_ids.take(post_pred.constant_data.home_idx.to_numpy())
    away_team = team_ids.take(post_pred.constant_data.away_idx.to_numpy())

    model_predictions = pd.DataFrame(data={c.HOME : home_team, c.AWAY : away_team, c.PREDICTION : win_probabilities})
    return model_predictions

def format_predictions_for_submission(pred_data: pd.DataFrame, season: int) -> pd.DataFrame:
    """Formats a set of predictions for submitting to the competion in ID, Pred format.

    Parameters
    ----------
    pred_data : pd.DataFrame
        A dataframe of model predictions with columns c.HOME, c.AWAY, c.PREDICTION.
    season : int
        The season to predict over.

    Returns
    -------
    pd.DataFrame
        A dataframe formatted for competition submission.
    """

    ids = []
    preds = []

    for _ , match in pred_data.iterrows():

        match_id = utils.format_match_id_from_teams(home=match[c.HOME], away=match[c.AWAY], season=season)
        match_pred = match[c.PREDICTION] if match[c.AWAY] > match[c.HOME] else 1 - match[c.PREDICTION]

        ids.append(match_id)
        preds.append(match_pred)

    output_df = pd.DataFrame(data={c.OUTPUT_ID : ids, c.OUTPUT_PRED : preds})
    return output_df