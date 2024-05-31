import logging
import sys
from pythonjsonlogger import jsonlogger


class JsonFormatter(jsonlogger.JsonFormatter):
    """Formatter for pretty-printing in Google's stackdriver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_log_record(self, log_record):
        """Overwrite JsonFormatter method."""
        log_record['severity'] = log_record['levelname']
        return log_record


def get_logger(name):

    logger = logging.getLogger(name)
    my_handler = logging.StreamHandler(sys.stdout)
    my_handler.setLevel(logging.DEBUG)
    my_formatter = JsonFormatter("%(asctime) %(filename) %(funcName) %(levelname) %(lineno) %(module) %(message) %(name) %(pathname) %(module)")
    my_handler.setFormatter(my_formatter)
    logger.addHandler(my_handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger
