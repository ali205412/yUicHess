"""
Helper methods for working with trained models.
"""

from logging import getLogger

logger = getLogger(__name__)
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_best_model_weight(model):
    """
    :param yui_chess.agent.model.ChessModel model:
    :return:
    """
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):
    """

    :param yui_chess.agent.model.ChessModel model:
    :return:
    """
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def reload_best_model_weight_if_changed(model):
    """

    :param yui_chess.agent.model.ChessModel model:
    :return:
    """
    if model.config.model.distributed:
        return load_best_model_weight(model)
    else:
        logger.debug("start reload the best model if changed")
        digest = model.fetch_digest(model.config.resource.model_best_weight_path)
        if digest != model.digest:
            return load_best_model_weight(model)

        logger.debug("the best model is not changed")
        return False
