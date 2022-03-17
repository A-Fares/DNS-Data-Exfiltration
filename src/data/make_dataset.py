import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from src.models.predict_model import predict_model
from kafka import KafkaConsumer
import warnings

warnings.filterwarnings('ignore')


## convert dataframe to csv files
def convert_record(result):
    # append data frame to CSV file
    return result.to_csv(r"E:\uOttawa\SecondTerm\AI for CS\Assignments_2\assignment2-A-Fares\data\final_output_dataset.csv", index=False)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    """
    run the training model to make pickle files for the saved model
    Args: training data csv file
    
    warning: try training by uncomment this line
    """
    ## get the training data from the csv to df
    # df = pd.read_csv(r"..\src\data\training_dataset.csv")

    ## run the kafka consumer
    consumer = KafkaConsumer(
        'ml-raw-dns',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=False
    )

    ## init result list, which appended the dataframe line by line
    result = []

    # Ingesting data from input topic
    for i, message in enumerate(consumer):
        if i < 100000:
            domain = message.value.decode("utf-8")
            result.append(predict_model(domain))
        else:
            break

    # concat the lists of dataframes to one dataframe
    result_df = pd.concat(result, ignore_index=True)
    convert_record(result_df)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
