from BrainTumorSegmentation import logger
from BrainTumorSegmentation.pipeline.stage_01_data_ingestion import (
    DataIngestionPipeline,
)

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"\n\n >>>>>>>>> The {STAGE_NAME} has started <<<<<<<<<<<< \n\n")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(
        f"\n\n >>>>>>>>> The {STAGE_NAME} has completed successful <<<<<<<<<<<< \n\n ========="
    )
except Exception as e:
    logger.exception(e)
    raise e
