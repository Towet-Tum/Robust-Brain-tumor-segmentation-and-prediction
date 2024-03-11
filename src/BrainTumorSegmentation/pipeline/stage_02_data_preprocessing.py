from BrainTumorSegmentation import logger
from BrainTumorSegmentation.config.configuration import ConfigurationManager
from BrainTumorSegmentation.components.data_preprocessing import DataPreprocess

STAGE_NAME = "Data Preprocessing"


class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_process_config = config.get_preprocess_config()
        data_process = DataPreprocess(config=data_process_config)
        data_process.resize_and_normalize()
        data_process.train_val_split()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n >>>>>>>>> The {STAGE_NAME} has started <<<<<<<<<<<< \n\n")
        data_preprocessing = DataPreprocessingPipeline()
        data_preprocessing.main()
        logger.info(
            f"\n\n >>>>>>>>> The {STAGE_NAME} has completed successful <<<<<<<<<<<< \n\n ========="
        )
    except Exception as e:
        logger.exception(e)
        raise e
