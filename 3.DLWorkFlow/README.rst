*******************************
DL WorkFlow
*******************************
- DL WorkFlow
	- Idea theory
	- Design Experiments
		- Dataloaders and datasets
		- Pretrained models (torchvision, torchtext and torchaudio)
		- Preprocess data/setting env
	- Implements model
		- Interfacing with environements (from soumith video, I suspect is with GYM)
			- any env with python api (show some example)
		- Saving checkpoints or intermediate stages
		- Debugging and profiling
			- bottleneck
			- profiler
	- Training & validation - splits again part of data loaders
		- Ignite
		- Distributed PyTorch
		- Building Optimizers and dealing with GPU
	- Publish and ship - covers in pytorch to production
		- Checkpointing
		- Loading and dumping models (storage.cude('device'))
		- Compilation and JIT
		- Saving model - equivalant to checkpointing in TF