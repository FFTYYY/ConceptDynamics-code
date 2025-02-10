echo "# ----- figures main paper -----"

echo "painting topology order"
python -m paint.topology_order --exp_id=1 

echo "painting curves main"
python -m paint.curve_multi --exp_id=2,3,4,5,6 --info=0.05,0.1,0.5,1,2 --device=cpu
python -m paint.curve_noise --exp_id=17,18,19 --info=1,2,3 --device=cpu
python -m paint.curve_multi --exp_id=28,29 --info="High dimensional,Low dimensional" --device=cpu   
python -m paint.loss_multi  --exp_id=28,29 --info="High dimensional,Low dimensional" --device=cpu

echo "painting theory"
cd figure\theory
python stages.py
cd ..\..

echo "painting theory triple"
cd figure\triple
python fail.py
python multiple.py
python triple.py
cd ..\..

echo "# ----- figures appendix -----"

echo "painting curves appendix"

python -m paint.curve_multi --exp_id=7,8,9,10,11 --info=0.05,0.1,0.5,1,2 --folder_prefix=appendix --device=cpu
python -m paint.curve_noise --exp_id=23,24,25 --info=1,2,3 --folder_prefix=appendix --device=cpu

python -m paint.curve_multi --exp_id=12,13,14,15,16 --info=0.05,0.1,0.5,1,2 --folder_prefix=appendix --device=cpu
python -m paint.curve_noise --exp_id=20,21,22 --info=1,2,3 --folder_prefix=appendix --device=cpu

python -m paint.curve_multi --exp_id=26 --info=""  --folder_prefix=appendix --device=cpu

python -m paint.curve_multi --exp_id=27 --info=""  --folder_prefix=appendix --device=cpu

python -m paint.loss_multi --exp_id=26,27 --info="Small $\mu$,Large $\mu$"  --folder_prefix=appendix --device=cpu
