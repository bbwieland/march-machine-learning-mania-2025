import model as mod
import predict as pred
import etl
import utils
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for the team-strength model.')
    parser.add_argument('--season', type=int, required=True, help='The season to run the model for')
    parser.add_argument('--league', type=str, required=True, help='The league to run the model for')
    args = parser.parse_args()

    league = args.league
    season = args.season

    # Preprocessing Data for Modeling
    df = etl.read_league_season_results(league=league, season=season)
    clean_df, team_ids = etl.preprocess_match_data(raw_data=df)
    data_dict, coords = etl.get_data_dict_and_coords(processed_data=clean_df, team_ids=team_ids)

    # Building and Fitting the Model
    game_model = mod.build_model(data_dict=data_dict, coords=coords)
    idata = mod.sample_model(model=game_model)

    # Parameter Analysis
    team_ratings = etl.extract_team_ratings_from_trace(trace=idata, league=league)

    # Predicting Potential Matchups
    post_pred_data = pred.create_posterior_team_pairings(team_ids=team_ids)
    post_pred = pred.predict_posterior(model=game_model, trace=idata, new_data=post_pred_data)

    # Formatting Predictions for Submission/Interpretability
    matchup_preds = pred.get_game_predictions_from_posterior(post_pred=post_pred, team_ids=team_ids)
    labeled_preds = etl.add_team_names_to_ids(pred_data=matchup_preds, league=league)
    output_preds = pred.format_predictions_for_submission(pred_data=matchup_preds)

    # Saving Outputs
    utils.save_team_ratings(ratings=team_ratings, league=league, season=season)
    utils.save_readable_predictions(labeled_preds=labeled_preds, league=league, season=season)
    utils.save_submission(submission=output_preds, league=league, season=season)