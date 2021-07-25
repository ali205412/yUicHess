import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
import numpy as np
from yui_chess.agent.model_chess import EngineModel
from yui_chess.config import Config
from yui_chess.env.chess_env import canon_input_planes, is_black_turn, testeval
from yui_chess.lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from yui_chess.lib.model_helper import load_best_model_weight
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras_adabound import AdaBound
logger = getLogger(__name__)
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def start(config: Config):
    return OptimizeWorker(config).start()


class OptimizeWorker:
   
    def __init__(self, config: Config):
        self.config = config
        self.model = None  
        self.dataset = deque(),deque(),deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        self.filenames = deque(get_game_data_filenames(self.config.resource))
        shuffle(self.filenames)
        total_steps = self.config.trainer.start_total_steps
        while True:
            self.fill_queue()
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            self.save_current_model()
            a, b, c = self.dataset
            while len(a) > self.config.trainer.dataset_size/2:
                a.popleft()
                b.popleft()
                c.popleft()

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02,
                             callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error'] 
        self.model.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def fill_queue(self):
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                logger.debug(f"loading data from {filename}")
                futures.append(executor.submit(load_data_from_file,filename))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                for x,y in zip(self.dataset,futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    logger.debug(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file,filename))

    def collect_all_loaded_data(self):
        state_ary,policy_ary,value_ary=self.dataset
        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        model = EngineModel(self.config)
        rc = self.config.resource
        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError("cant load nthn")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading newsest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    return convert_to_cheating_data(data)


def convert_to_cheating_data(data):
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:
        state_planes = canon_input_planes(state_fen)
        if is_black_turn(state_fen):
            policy = Config.flip_policy(policy)
        move_number = int(state_fen.split(' ')[5])
        value_certainty = min(5, move_number)/5
        sl_value = value*value_certainty + testeval(state_fen, False)*(1-value_certainty)
        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)
    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list, dtype=np.float32)
