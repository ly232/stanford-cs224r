# General notes



## Tabular vs. Deep RL

|Component     |Silver/Classical RL          |CS224R / Deep RL                         |
|--------------|-----------------------------|-----------------------------------------|
|Storage       |A Table (Entry for every s)  |MLP Weights (θ)                          |
|Generalization|None (States are independent)|High (Similar states s produce similar a)|
|Improvement   |π(s)=argmaxa​Q(s,a)          |θ←θ+α∇θ​J(θ) (Policy Gradient)           |
|The "Actor"   |The current π in the table   |The MLP parameterized by θ               |

Generalized Policy Iteration vs. Deep RL:
* Policy evaluation: update V(s) by Critic MLP to minimize Bellman error.
* Policy improvement: update π(.) by gradient ascent over the Actor MLP to maximize the reward.


## Neuro net training

See `cs224r.policies.MLP_policy.MLPPolicySL`, and utils `cs224r.infrastructure.pytorch_util.ptu.build_mlp`.

* `build_mlp`: scaffold the newtork with input size, hidden layer sizes, output size, and activation layers in between.
* `MLPPolicySL.forward`: takes observation tensors to run one forward pass.
* `MLPPolicySL.update`: forward prop, compute loss, backward prop.

Big picture is to think of `MLPPolicySL` as the core of policy improvement step in generalized policy iteration (GPI).
