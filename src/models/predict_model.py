import pickle
import numpy as np
import pandas as pd
from src.features.build_features import build_features, preprocessing_data

## Load the model
with open(r"..\models\model_rf.sav", 'rb') as pickle_file:
    rf = pickle.load(pickle_file)


## predict the new features
def predict_model(domain):
    """
    returns a dataframe with all the columns created and populated

    Args:
        str : a value of the DNS data from the consumer

    Returns:
        A dataframe.
    """
    domain_df = ['' + domain]
    domain_df = pd.DataFrame(domain_df, columns=['domain'])
    generated_features_df = build_features(domain_df)

    preprocess_df = generated_features_df.copy()
    preprocess_df = preprocessing_data(preprocess_df)
    # Find the prediction score and confidence score of the model
    pred_score = rf.predict(preprocess_df)
    confidence_score = np.round(rf.predict_proba(preprocess_df)[0][pred_score], 2)

    data2 = {
        'domain': domain,
        'predicted_label': pred_score,
        'score': confidence_score
    }
    pred_df = pd.DataFrame(data2)
    result_df = pd.concat([generated_features_df, pred_df], axis=1)

    return result_df
