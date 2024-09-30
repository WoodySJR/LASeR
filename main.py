import random
import time
import numpy as np
from LLM.run_llm import run
from ppo.arguments import get_args

import torch

if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    args = get_args()
    args.num_processes = 4
    args.cuda = False
    args.no_cuda = True
    args.eval_interval = 50
    tasks = ["Carrier-v0"]
    
    for task in tasks:
    
        run(
            pop_size = 25,
            structure_shape = (5,5),
            experiment_name = task + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())),
            max_evaluations = 1000,
            train_iters = 1000,
            num_cores = 25,
            env_name = task, 
            args = args
        )