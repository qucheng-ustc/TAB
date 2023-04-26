import functools
import logging

@functools.cache
def get_logger(name='arrl',
            level='INFO',
            fmt = '%(asctime)s---%(name)s---%(levelname)s---%(message)s',
            file_name='logs/exp.log'):
    logger = logging.getLogger(name)
 
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    had_fmt = logging.Formatter(fmt=fmt)
    handler.setFormatter(had_fmt)
    logger.addHandler(handler)
 
    file_handler = logging.FileHandler(file_name, encoding='utf-8')
    file_handler.setLevel(level)
    fh_fmt = logging.Formatter(fmt=fmt)
    file_handler.setFormatter(fh_fmt)
    logger.addHandler(file_handler)
    def logger_print(*values, sep=" "):
        logger.info(sep.join([str(v) for v in values]))
    logger.print = logger_print
    return logger
