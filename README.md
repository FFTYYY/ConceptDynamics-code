## Code for "Dynamics of Concept Learning and Compositional Generalization"

This repo contains the code for the ICLR 2025 paper [Dynamics of Concept Learning and Compositional Generalization](https://arxiv.org/abs/2410.08309).

------
Before running the code, please make sure [xingyun](https://github.com/FFTYYY/XingYun) is installed.
```
pip install xingyun
```
`xingyun` has been set to store data locally so no further configuration is required.

------
To reproduce all the figures of the SIM task in the paper, run the following command. 
```
source run.sh   # .\run.bat on windows
source paint.sh # .\paint.bat on windows
```
All figures will be saved in `figure/`.
Please see `run.sh` for the detailed commands.

------
To reproduce the figures of the diffusion task, run 
```
python -m paint.curve_diffusion
```
The figures will be saved in `figure/diffusion/`.
For more details with the diffusion task, please refer to [this repo](https://github.com/cfpark00/concept-learning).

