import model as mod
import predict as pred
import etl
import utils
import argparse
import logging
from typing import Literal, Dict
import pandas as pd

from constants import LEAGUES

logger = logging.getLogger(__name__)
logging.basicConfig(filename='model_pipeline.log', encoding='utf-8', level=logging.INFO)

def run_full_model_pipeline(league: Literal['M', 'W'], season: int) -> Dict[str, pd.DataFrame]:
    """High-level wrapper for CLI interfacing. Fits the full model with all outputs for the given parameters.

    Parameters
    ----------
    league : Literal['M', 'W']
        Whether to fit the model for men's or women's NCAA basketball.
    season : int
        The season to model.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing model outputs: 
        - 'ratings' : a dataframe of team ratings.
        - 'predictions' : a readable labeled dataframe of matchup predictions.
        - 'submission' : a dataframe of predictions formatted for competition submission.
    """
    # Preprocessing Data for Modeling
    df = etl.read_league_season_results(league=league, season=season)
    logger.info("Imported data.")
    clean_df, team_ids = etl.preprocess_match_data(raw_data=df)
    logger.info("Preprocessed data.")
    data_dict, coords = etl.get_data_dict_and_coords(processed_data=clean_df, team_ids=team_ids)
    logger.info("Formatted data for PyMC model ingestion.")

    # Building and Fitting the Model
    logger.info("Beginning sampling process...")
    game_model = mod.build_model(data_dict=data_dict, coords=coords)
    idata = mod.sample_model(model=game_model)
    logger.info("Completed model sampling!")

    # Parameter Analysis
    team_ratings = etl.extract_team_ratings_from_trace(trace=idata, league=league)

    # Predicting Potential Matchups
    logger.info("Beginning prediction process...")
    post_pred_data = pred.create_posterior_team_pairings(team_ids=team_ids)
    post_pred = pred.predict_posterior(model=game_model, trace=idata, new_data=post_pred_data)
    logger.info("Made predictions!")

    # Formatting Predictions for Submission/Interpretability
    logger.info("Formatting predictions for submission...")
    matchup_preds = pred.get_game_predictions_from_posterior(post_pred=post_pred, team_ids=team_ids)
    labeled_preds = etl.add_team_names_to_ids(pred_data=matchup_preds, league=league)
    output_preds = pred.format_predictions_for_submission(pred_data=matchup_preds)

    return {'ratings' : team_ratings, 'predictions' : labeled_preds, 'submission' : output_preds}

def save_predictions(pred_dict: Dict[str, pd.DataFrame], league: Literal['M', 'W'], season: int):
    """Given a dictionary from `run_full_model_pipeline`, saves necessary data to csvs.

    Parameters
    ----------
    pred_dict : Dict[str, pd.DataFrame]
        The dictionary of model team ratings, predictions, and a submission dataframe.
    league : Literal['M', 'W']
        Whether to fit the model for men's or women's NCAA basketball.
    season : int
        The season to model.
    """

    ratings = pred_dict['ratings']
    predictions = pred_dict['predictions']
    submission = pred_dict['submission']

    utils.save_team_ratings(ratings=ratings, league=league, season=season)
    utils.save_readable_predictions(labeled_preds=predictions, league=league, season=season)
    utils.save_submission(submission=submission, league=league, season=season)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for the team-strength model.')
    parser.add_argument('--season', type=int, required=True, help='The season to run the model for')
    args = parser.parse_args()
    season = args.season

    m_outputs = run_full_model_pipeline(league="M", season=season)
    w_outputs = run_full_model_pipeline(league="W", season=season)

    save_predictions(m_outputs, league="M", season=season)
    save_predictions(w_outputs, league="W", season=season)

    utils.combine_and_save_full_submission(m_preds=m_outputs['submission'], w_preds=w_outputs['submission'], season=season)
    logger.info(f"Successfully executed {__name__}!")