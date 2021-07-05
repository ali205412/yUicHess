# yUicHess
A Reinforcement Learning Chess Engine ;)
- **Reinforcement Learning Modules**  
	- The engine is made up of three workers: `self`, `opt` and `eval`.
		- self is the self play module that utilizes the best model trained to generate new games to train the model.
		- opt is the training module that trains the model, and generates new models to be evaluated.
		- eval is the evaluation module that compares the newley generated model to the current model and replaces best model if it is proven to be inferior

- **GUI module**	

	- the gui module allows the engine to communicate with gameboards throughout the UCI protocol (Universal Chess Interface)
	- also allows the engine to communicate with a chess interface - all that is needed is to point it to yUiUCI.bat

- **Setup**	

	1. install Anaconda3 python3.8
	2. install my anaconda env `chessengine.yml`
	3. install [Cuda 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) to utilize GPU accelerated training
	4. adjust the .bat files replacing `ali20` with your system's name, for example `C:\Users\[system_name]\anaconda3\Scripts\`

- **Use**	
 	- In order to train the model run self, opt and eval functions.
 	- Adjust settings used in normal config file in order to further tailor the training process to the equipment you are using

- **Recommended Equipment**	

	- I generated self play games for 200 hours,  ran the opt function for all generated play data, and then used the eval function to evaluate models generated in order to find best model. 

	- All was done on a K80 gpu using GCP (Google Cloud Platform).

	- Any CPU could be used, but I recommend at least a 4 core CPU with a minimum clockspeed of 4 Ghz, and the system memory should at least be 32 GB's to avoid any bottlenecks.

- [**Resesearch on Background and Related works, and related technologies**](https://docs.google.com/document/d/14dOU6QFc-1rZ_3eqg3ifCOgDc-JjadE_rQyrVHZNdMU/edit?usp=sharing)

- [**Project process log and engine breakdown and analysis**](https://docs.google.com/presentation/d/11OXL5jcayGdL1V-T7D2m4rcXQ_JtFJcp3NWrGJXBp6A/edit?usp=sharing)

- [**Project Gantt Chart**](https://docs.google.com/spreadsheets/d/1DUmkcPceNDXtFXZ1pDVqOrEvx1-5kkgY-qdOPaMNvuQ/edit?usp=sharing)


- **Resources used**
	- [Surag's Alpha Zero Implementation](https://web.stanford.edu/~surag/posts/alphazero.html)
	- [Deepminds Alpha Zero config](https://kstatic.googleusercontent.com/files/2f51b2a749a284c2e2dfa13911da965f4855092a179469aedd15fbe4efe8f8cbf9c515ef83ac03a6515fa990e6f85fd827dcd477845e806f23a17845072dc7bd) played a huge role as inspiration for the initial settings used.
	- [Zeta36's AlphaZero Implementation](https://github.com/Zeta36/chess-alpha-zero) served a huge role as inspiration for the structure and design of the engine, however the model, weights and settings used are independent of those chosen by Zeta36
