# Includes element that realate to direct gameplay, in addition to returning prediction values
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock

import chess
import numpy as np

from yui_chess.config import Config
from yui_chess.env.chess_env import ChessEnv, Winner

logger = getLogger(__name__)


class VisitStats:

    def __init__(self):
        #deafultdict represents all possible moves that could be done from the
        #current board state
        self.a = defaultdict(ActionStats)
        #tallys number of visits that have been done
        self.sum_n = 0


class ActionStats:

    def __init__(self):
        #n. of times potential moves is visited by the MCTS
        self.n = 0
        #n. of times a subsect of the move is visited by the MCTS
        self.w = 0
        #move rating (x bar) (total rating / by n.of visited)
        self.q = 0
        #r value of attempting to engage in the move
        self.p = 0


class ChessPlayer:

    def __init__(self, config: Config, pipes=None, play_config=None, dummy=False):
        self.moves = []
        #Keeps a record of all moves that have occured during the coourse of the game
        self.tree = defaultdict(VisitStats)
        self.config = config
        #Temporarily stores the settings chosen from the options.py file in RAM
        self.play_config = play_config or self.config.play
        self.labels_n = config.n_labels
        #Stores all potential move in label form e.x. h1g1
        self.labels = config.labels
        #links the index with the labels
        self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.labels_n))}
        if dummy:
            return

        self.pipe_pool = pipes
        #pipe relays the current game state in order to be able to predict the policy and value outputs
        self.node_lock = defaultdict(Lock)
        #Encountered error in multi threaded predicitons, thus created to log wether a game state is being explored by another thread
        #All movements along the tree are stored
    def reset(self):

        self.tree = defaultdict(VisitStats)
        #resets VisitStats in order to initiate a new journey

    def deboog(self, env):
        print(env.testeval())

        state = state_key(env)
        my_visit_stats = self.tree[state]
        stats = []
        for action, a_s in my_visit_stats.a.items():
            moi = self.move_lookup[action]
            stats.append(np.asarray([a_s.n, a_s.w, a_s.q, a_s.p, moi]))
        stats = np.asarray(stats)
        a = stats[stats[:,0].argsort()[::-1]]

        for s in a:
            print(f'{self.labels[int(s[4])]:5}: '
                  f'n: {s[0]:3.0f} '
                  f'w: {s[1]:7.3f} '
                  f'q: {s[2]:7.3f} '
                  f'p: {s[3]:7.5f}')

    def action(self, env, can_stop = True) -> str:
        #Function serves to try and predict the best potential action that could be taken in order to maximize winning potential
        self.reset()

        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.labels_n), p = self.apply_temperature(policy, env.num_halfmoves)))
        #Whether or not the chosen action is legal
        if can_stop and self.play_config.resign_threshold is not None and \
                        root_value <= self.play_config.resign_threshold \
                        and env.num_halfmoves > self.play_config.min_resign_turn:
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[my_action]

    def search_moves(self, env) -> (float, float):
        #Scouts all explored game states and returns the game state with the greatest rating/value.
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0] 

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
        #Some possible moves are explored then interated into a list, when then filter through and chooses the best move the was found during the exploration.
        if env.done:
            if env.winner == Winner.draw:
                return 0
           
            return -1

        state = state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v

            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env) 
        leaf_v = -leaf_v


        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):
        #A new leaf is expanded which gets a prediction for its policy and value output in the current game state.
        state_planes = env.canonical_input_planes()

        leaf_p, leaf_v = self.predict(state_planes)

        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p)

        return leaf_p, leaf_v

    def predict(self, state_planes):
        #Recieves a output for the policy and value
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        #Observation state is represented in the form of a plane
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret
        #Returns the policy value hence the prior probability of taking the action leading to this state and value value (sounds confusing :|) which is the value of the state prediction for this specific state.

    def select_action_q_and_u(self, env, is_root_node) -> chess.Move:
        #Chooses which tree branch should be explored through prioritizing high value pathways
        state = state_key(env)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None:
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1) 

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))
        
        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn):
        # A random variation to the chance of picking a certain move 
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        # Returns probaility of choosing all potential moves through taking into account the number of times it is visited.
        state = state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def sl_action(self, observation, my_action, weight=1):
        #Logs all moves in self.moves, in order to generate game play data
        policy = np.zeros(self.labels_n)

        k = self.move_lookup[chess.Move.from_uci(my_action)]
        policy[k] = weight

        self.moves.append([observation, list(policy)])
        return my_action

    def finish_game(self, z):
        #when termination occurs all values of previous moves are updated
        for move in self.moves:  
            move += [z]


def state_key(env: ChessEnv) -> str:

    fen = env.board.fen().rsplit(' ', 1) 
    return fen[0]
