import logging

def init_preprocess_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='logs/preprocessing.log',
        level=logging.INFO,
        style="{",
        format="{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M",
        encoding='utf-8',
    )
    return logger