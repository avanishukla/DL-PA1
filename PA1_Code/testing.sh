python3 train.py --lr 0.009 --momentum 0.5 --num_hidden 1 --sizes 1100 --activation tanh --loss ce --opt adam --epochs 25 --batch_size 300 --anneal true  --pretrain True --state 1 --testing True --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --test ../save_dir/best/test_1.csv
python3 train.py --lr 0.009 --momentum 0.5 --num_hidden 1 --sizes 1000 --activation tanh --loss ce --opt adam --epochs 25 --batch_size 200 --anneal true  --pretrain True --state 2 --testing True --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --test ../save_dir/best/test_2.csv