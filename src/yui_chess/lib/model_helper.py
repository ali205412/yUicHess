
from logging import getLogger

logger = getLogger(__name__)
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_best_model_weight(model):

    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):

    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def reload_best_model_weight_if_changed(model):

    if model.config.model.distributed:
        return load_best_model_weight(model)
    else:
        logger.debug("begin to refeed the aightest model if tampered with")
        digest = model.fetch_digest(model.config.resource.model_best_weight_path)
        if digest != model.digest:
            return load_best_model_weight(model)

        logger.debug("aighest model stil aightest model")
        return False
