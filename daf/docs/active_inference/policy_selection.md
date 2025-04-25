# Policy Selection in Active Inference

This document explains how policy selection works within the Active Inference framework for the FlyBody environment.

## Overview

In Active Inference, policy selection is based on minimizing expected free energy (EFE) rather than maximizing expected rewards as in traditional reinforcement learning. This approach unifies exploration and exploitation within a single objective function and accounts for both goal-directed behavior and information-seeking.

## Key Components

### 1. Policy Representation

Policies in Active Inference are temporal sequences of actions:

- **Finite horizon**: Typically planning 5-20 steps ahead
- **Discrete vs. continuous**: Can handle both action spaces
- **Hierarchical policies**: Different temporal scales (for complex behaviors)

For FlyBody, we'll support:
- Fine-grained motor control (continuous action space)
- Varying planning horizons based on task complexity
- Compositional policies for complex behaviors

### 2. Expected Free Energy

The core of policy selection in Active Inference is the calculation of expected free energy (EFE):

```
EFE(π) = ∑_τ [ Ambiguity + Risk + Complexity ]
```

Where:
- **Ambiguity**: How uncertain observations are given predicted states
- **Risk**: How far predicted states are from preferred (goal) states
- **Complexity**: KL divergence between predicted and prior states

In code, this looks like:

```python
def calculate_expected_free_energy(self, policy, current_belief, goal_prior):
    efe = 0
    belief = current_belief
    
    for action in policy:
        # Predict next belief after action
        next_belief = self.transition_model.predict(belief, action)
        
        # Expected observations
        expected_obs = self.observation_model.predict(next_belief)
        
        # Ambiguity: uncertainty in observations
        ambiguity = self.observation_model.entropy(next_belief)
        
        # Risk: divergence from preferred states
        risk = kl_divergence(next_belief, goal_prior)
        
        # Complexity: divergence from prior beliefs
        complexity = kl_divergence(next_belief, self.prior_belief)
        
        efe += ambiguity + risk + complexity
        belief = next_belief
        
    return efe
```

### 3. Policy Selection Algorithm

The policy selection process:

1. **Generate candidate policies**
   - For continuous action spaces: sample using MPPI or CEM
   - For discrete actions: evaluate all possible action sequences

2. **Evaluate EFE for each policy**
   - Simulate future beliefs and observations
   - Calculate expected free energy components
   
3. **Soft selection using precision-weighted distribution**
   - Higher precision = more deterministic selection
   - Lower precision = more exploratory behavior

```python
def select_policy(self, current_belief, goal_prior, num_candidates=100):
    # Generate candidate policies
    candidate_policies = self.generate_policies(num_candidates)
    
    # Calculate EFE for each policy
    efes = [self.calculate_expected_free_energy(p, current_belief, goal_prior) 
            for p in candidate_policies]
    
    # Convert to policy distribution using softmax with precision parameter
    policy_probs = softmax(-self.precision * np.array(efes))
    
    # Sample from the distribution or take the MAP estimate
    if self.deterministic:
        selected_idx = np.argmax(policy_probs)
    else:
        selected_idx = np.random.choice(len(candidate_policies), p=policy_probs)
    
    return candidate_policies[selected_idx]
```

### 4. Integration with FlyBody Control

The policy selection module interfaces with FlyBody's action space by:

1. Translating high-level policies to low-level motor commands
2. Handling the continuous nature of flight/walk control
3. Managing temporal aspects of motor execution

## Implementation Details

### Instantiating the Policy Selection Module

```python
from flybody.agents.active_inference import PolicySelection

policy_selection = PolicySelection(
    state_dim=perception.hidden_state_dim,
    action_dim=env.action_space.shape[0],
    horizon=10,
    num_candidates=200,
    precision=5.0
)
```

### Key Methods

```python
# Generate and evaluate policies
selected_policy = policy_selection.select(current_belief, goal_prior)

# Get the immediate action from the selected policy
action = selected_policy.get_current_action()

# Update the internal models based on experience
policy_selection.update(previous_belief, action, current_belief, reward)
```

### Adaptive Planning

The policy selection module can adapt its planning strategies:

- **Dynamic horizon**: Extend planning horizon in uncertain situations
- **Precision modulation**: Adjust exploration vs. exploitation based on task progress
- **Hierarchical planning**: Use abstractions for long-term planning, details for immediate actions

## Example: Walking Task Policy Selection

For a walking task, the policy selection would:

1. Generate candidate walking gaits and movement trajectories
2. Evaluate each based on:
   - Energy efficiency
   - Stability maintenance
   - Progress toward the goal
   - Information gain about uncertain terrain
3. Select policies that balance goal-directed locomotion with safety and efficient information gathering
4. Adapt gait patterns based on terrain features inferred by the perception module

### Visualization and Logging

The policy selection module provides methods to visualize:

- Policy distributions and selection probabilities
- Expected free energy components for different policies
- Simulated trajectories for candidate policies
- Evolution of planning horizons and precision parameters 