# Acts as a relay attaining the game state and returning the values outputted from
# the policy and value networks
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

class modelInterface:
   
    def __init__(self, computeModel):  
       
        self.computeModel = computeModel
        self.pipes = []

    def start(self):
        # Monitors and relays value on pipe to return predicted values
        projectionOperator = Thread(target=self.projectionCluster, name="projectionOperator")
        projectionOperator.daemon = True
        projectionOperator.start()

    def create_pipe(self):
       # Multi-Direction pipe
        moi, tu = Pipe()
        self.pipes.append(moi)
        return tu

    def projectionCluster(self):
        # Monitors self.pipes and relays any policy or value network outputs
        global graph
        graph = tf.compat.v1.get_default_graph()

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
                policy_ary, value_ary = self.computeModel.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))

