# 2026-02-21

Behavior cloning against Ant-v4 environment.

```
(.venv) ly232@ly232s-iMac-Pro hw1 % python cs224r/scripts/run_hw1.py \
        --expert_policy_file cs224r/policies/experts/Ant.pkl \
        --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
        --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
        --video_log_freq -1
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
Loading expert policy from... cs224r/policies/experts/Ant.pkl
obs (1, 111) (1, 111)
Done restoring expert policy...
########################
Logging outputs to  /Users/ly232/github/stanford-cs224r/hw1/cs224r/scripts/../../data/q1_bc_ant_Ant-v4_21-02-2026_20-34-58
########################
PyTorch detects an Apple GPU: running on MPS
/Users/ly232/github/stanford-cs224r/hw1/.venv/lib/python3.11/site-packages/gym/core.py:317: DeprecationWarning: WARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
  deprecation(
/Users/ly232/github/stanford-cs224r/hw1/.venv/lib/python3.11/site-packages/gym/wrappers/step_api_compatibility.py:39: DeprecationWarning: WARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
  deprecation(


********** Iteration 0 ************

Collecting data to be used for training iteration 0...

Training agent using sampled data from replay buffer...

Beginning logging procedure...

Collecting data for eval...
/Users/ly232/github/stanford-cs224r/hw1/.venv/lib/python3.11/site-packages/gym/utils/passive_env_checker.py:241: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
  if not isinstance(terminated, (bool, np.bool8)):

Saving rollouts as videos...
Eval_AverageReturn : 4582.0537109375
Eval_StdReturn : 0.0
Eval_MaxReturn : 4582.0537109375
Eval_MinReturn : 4582.0537109375
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 4713.6533203125
Train_StdReturn : 12.196533203125
Train_MaxReturn : 4725.849609375
Train_MinReturn : 4701.45654296875
Train_AverageEpLen : 1000.0
Train_EnvstepsSoFar : 2000
TimeSinceStart : 21.045540809631348
Training Loss : -15.763310432434082
Initial_DataCollection_AverageReturn : 4713.6533203125
Done logging...
```

# 2026-02-22

Add more `--eval_batch_size` than `--ep_len` to collect more means and stdevs per iteration. Note for behavior cloning, we only see expert data in one single batch, so in the end there's still just a single mean and single stdev on tensorboard. DAGGER would allow for more iterations (setting `--n_iter` larger than 1 would hit a validation failure if it's running in BC mode).

```
(.venv) ly232@ly232s-iMac-Pro hw1 % python cs224r/scripts/run_hw1.py \
        --expert_policy_file cs224r/policies/experts/Ant.pkl \
        --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
        --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
        --ep_len 1000 \
        --eval_batch_size 5000 \
        --video_log_freq 1
```

Open tensorboard:

```
python -m tensorboard.main --logdir data/
```

Eval (student):
![Eval rollout](screenshots/eval_rollout_20260222.gif)

Train (expert):
![Trained rollout](screenshots/train_rollout_20260222.gif)
