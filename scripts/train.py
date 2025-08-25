"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Using 'weights' as positional parameter"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Arguments other than a weight enum or `None` for 'weights'"
)

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach
from training.distill_coach import DistillCoach

def main():
	opts = TrainOptions().parse()
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)
	if not hasattr(opts, 'latent_distill_lambda'):
		opts.latent_distill_lambda = 1.0
	if not hasattr(opts, 'feature_distill_lambda'):
		opts.feature_distill_lambda = 0.5 

	coach = Coach(opts)
	
	#coach = DistillCoach(opts)
	coach.train()
	


if __name__ == '__main__':
	main()
