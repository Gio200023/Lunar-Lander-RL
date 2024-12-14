# Implementation of REINFORCE and Actor-Critic using Lunar Lander

To replicate the experiments, you can run the file `Experiment.py`

To run the REINFORCE algorithm, run:
```
python Experiment.py --reinforce
```

To run Actor-Critic with baseline and bootstrap, run:
```
python Experiment.py --actor_critic
```

To run Actor-Critic with just baseline, run:

```
python Experiment.py --ac_base 
```

To run Actor-Critic with just bootstrap, run:

```
python Experiment.py --ac_boot
```

To run them together, run:
```
python Experiment.py --ac_boot --ac_base --actor_critic --reinforce
```
