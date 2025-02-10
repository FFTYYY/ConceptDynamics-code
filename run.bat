python main.py --id=1  --data=multi_identity --sigma=0.5 
python main.py --id=2  --data/dists=1,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=1
python main.py --id=3  --data/dists=1,2 --data/sigma_lb=0.1,0.05 --data/sigma_rb=0.1,0.05 --data/sigma_lt=0.1,0.05 --model/n_layers=1
python main.py --id=4  --data/dists=1,2 --data/sigma_lb=0.5,0.05 --data/sigma_rb=0.5,0.05 --data/sigma_lt=0.5,0.05 --model/n_layers=1
python main.py --id=5  --data/dists=1,2 --data/sigma_lb=1,0.05 --data/sigma_rb=1,0.05 --data/sigma_lt=1,0.05 --model/n_layers=1
python main.py --id=6  --data/dists=1,2 --data/sigma_lb=2,0.05 --data/sigma_rb=2,0.05 --data/sigma_lt=2,0.05 --model/n_layers=1
python main.py --id=7  --data/dists=1,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2
python main.py --id=8  --data/dists=1,2 --data/sigma_lb=0.1,0.05 --data/sigma_rb=0.1,0.05 --data/sigma_lt=0.1,0.05 --model/n_layers=2
python main.py --id=9  --data/dists=1,2 --data/sigma_lb=0.5,0.05 --data/sigma_rb=0.5,0.05 --data/sigma_lt=0.5,0.05 --model/n_layers=2
python main.py --id=10 --data/dists=1,2 --data/sigma_lb=1,0.05 --data/sigma_rb=1,0.05 --data/sigma_lt=1,0.05 --model/n_layers=2
python main.py --id=11 --data/dists=1,2 --data/sigma_lb=2,0.05 --data/sigma_rb=2,0.05 --data/sigma_lt=2,0.05 --model/n_layers=2
python main.py --id=12 --data/dists=1,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2 --relu
python main.py --id=13 --data/dists=1,2 --data/sigma_lb=0.1,0.05 --data/sigma_rb=0.1,0.05 --data/sigma_lt=0.1,0.05 --model/n_layers=2 --relu
python main.py --id=14 --data/dists=1,2 --data/sigma_lb=0.5,0.05 --data/sigma_rb=0.5,0.05 --data/sigma_lt=0.5,0.05 --model/n_layers=2 --relu
python main.py --id=15 --data/dists=1,2 --data/sigma_lb=1,0.05 --data/sigma_rb=1,0.05 --data/sigma_lt=1,0.05 --model/n_layers=2 --relu
python main.py --id=16 --data/dists=1,2 --data/sigma_lb=2,0.05 --data/sigma_rb=2,0.05 --data/sigma_lt=2,0.05 --model/n_layers=2 --relu
python main.py --id=17 --data/dists=1,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=1
python main.py --id=18 --data/dists=2,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=1
python main.py --id=19 --data/dists=3,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=1
python main.py --id=20 --data/dists=1,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2 --relu
python main.py --id=21 --data/dists=2,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2 --relu
python main.py --id=22 --data/dists=3,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2 --relu
python main.py --id=23 --data/dists=1,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2
python main.py --id=24 --data/dists=2,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2
python main.py --id=25 --data/dists=3,2 --data/sigma_lb=0.05,0.05 --data/sigma_rb=0.05,0.05 --data/sigma_lt=0.05,0.05 --model/n_layers=2
python main.py --id=26 --no_rotation --lr=0.01 --dim=3 --hdim=3 --n_epochs=40 --data/dists=1,2 --model/n_layers=3
python main.py --id=27 --no_rotation --lr=0.01 --dim=3 --hdim=3 --n_epochs=40 --data/dists=2,4 --model/n_layers=3
python main.py --id=28 --lr=0.1 --n_epochs=40 --data/dists=2,4 --model/n_layers=4
python main.py --id=29 --model/n_layers=4 --data/dists=2,4 --dim=2 --hdim=4 --lr=0.01