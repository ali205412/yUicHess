
from multiprocessing import connection, Pipe
from threading import Thread
import tensorflow as tf
import numpy as np
from keras import backend as K

from yui_chess.config import Config
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

class ChessModelAPI:
   
    def __init__(self, agent_model):  # ChessModel
       
        self.agent_model = agent_model
        self.pipes = []

    def start(self):
        
        prediction_worker = Thread(target=self._predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def create_pipe(self):
       
        me, you = Pipe()
        self.pipes.append(me)
        return you

    def _predict_batch_worker(self):
        
        global graph
        graph = tf.get_default_graph()

        while True:
            ready = connection.wait(self.pipes,timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            with graph.as_default():
                policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))

