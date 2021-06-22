# yUicHess
A Reinforcement Learning Chess Engine ;)
Reinforcement Learning Modules
  The eninge is made up of three workers: self, opt and eval.
    self is the self play module that utilizes the best model trained to generate new games to train the model.
    opt is training module that trains the model, and generates new models to be evaluated.
    eval is the evaluation module that compares the newley generated model to the current model and replaces best model if it is proven to be inferior

GUI module
	the gui module allows the engine to communicate with gameboards throught the UCI protocol (Universal Chess Interface)
	to allow the engine to communicate with a chess interface all that is needed is to point it to yUiUCI.bat
