# yUicHess
A Reinforcement Learning Chess Engine ;)
- **Reinforcement Learning Modules**  
	- The eninge is made up of three workers: `self`, `opt` and `eval`.
		- self is the self play module that utilizes the best model trained to generate new games to train the model.
		- opt is training module that trains the model, and generates new models to be evaluated.
		- eval is the evaluation module that compares the newley generated model to the current model and replaces best model if it is proven to be inferior

-**GUI module**	

-the gui module allows the engine to communicate with gameboards throught the UCI protocol (Universal Chess Interface)
-to allow the engine to communicate with a chess interface all that is needed is to point it to yUiUCI.bat

-**Setup**	

1. install Anaconda3 python3.8
2. install my anaconda env
3. install [Cuda 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) to utilize GPU accelerated training
4. adjust the .bat files replacing ali20 with your systems's name for example `C:\Users\[system_name]\anaconda3\Scripts\`

-**Use**	

In order to train the model run self, opt and eval.
settings used in normal config file in order to further tailor the training process to the equipment you are using

-**Recommended Equipment**	

I generated self play games for 200 hours,  ran the opt function for all generated play data, and then used the eval function to evaluate models generated in order to find best model. 
All was done on a K80 gpu using GCP (Google Cloud Platform).
Any CPU could be used but I recommend at least a 4 core CPU with a minimum clockspeed of 4 Ghz, and the system memory should at least be 32 GB's to avoid any bottlenecks.
