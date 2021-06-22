# yUicHess
A Reinforcement Learning Chess Engine ;)
Reinforcement Learning Modules
  The eninge is made up of three workers: self, opt and eval.
    self is Self-Play to generate training data by self-play using BestModel.
    opt is Trainer to train model, and generate next-generation models.
    eval is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.
