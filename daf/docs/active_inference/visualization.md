# Visualization and Logging for Active Inference

This document details how to visualize, log, and analyze the internal variables and processes of an Active Inference agent in the FlyBody framework.

## Overview

Comprehensive visualization and logging are essential for understanding Active Inference agents because:

1. The internal belief states are critical for understanding behavior
2. Expected free energy calculations drive policy selection
3. The perception-action cycle involves multiple interacting components
4. Precision parameters modulate exploration vs. exploitation

## Key Visualizable Components

### 1. Belief States

Belief states represent the agent's current understanding of the world:

- **Hidden state distributions**: Mean and variance of estimated states
- **Prediction errors**: Difference between predicted and actual observations
- **Free energy**: Value being minimized during perception
- **Uncertainty maps**: For spatial tasks, visualizing uncertainty across space

### 2. Policy Evaluation

Policy selection involves comparing multiple possible action sequences:

- **Policy distribution**: Probabilities assigned to each candidate policy
- **Expected free energy components**: 
  - Ambiguity (epistemic value)
  - Risk (pragmatic value) 
  - Complexity
- **Simulated trajectories**: Predicted paths for different policies
- **Selected actions**: The final chosen actions over time

### 3. Learning Process

The evolution of the agent's models:

- **Model parameters**: Weights of neural networks or parameters of other models
- **Learning curves**: How quickly the agent adapts to new environments
- **Generalization**: Performance on unseen scenarios
- **Adaptation**: Changes in policy selection with changing environments

## Implementation

### Logging Framework

The Active Inference agent uses TensorBoard for logging:

```python
class ActiveInferenceLogger:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step_counter = 0
        self.episode_counter = 0
        
    def log_belief_state(self, belief_state, step=None):
        """Log belief state variables to TensorBoard."""
        step = step or self.step_counter
        with self.writer.as_default():
            # Log means and variances of belief states
            for i, (mean, var) in enumerate(zip(belief_state.mean, belief_state.variance)):
                tf.summary.scalar(f'belief/mean/{i}', mean, step=step)
                tf.summary.scalar(f'belief/variance/{i}', var, step=step)
            
            # Log overall uncertainty
            tf.summary.scalar('belief/total_uncertainty', 
                             np.sum(belief_state.variance), step=step)
    
    def log_free_energy(self, free_energy, step=None):
        """Log free energy components to TensorBoard."""
        step = step or self.step_counter
        with self.writer.as_default():
            tf.summary.scalar('free_energy/total', free_energy['total'], step=step)
            tf.summary.scalar('free_energy/accuracy', free_energy['accuracy'], step=step)
            tf.summary.scalar('free_energy/complexity', free_energy['complexity'], step=step)
    
    def log_policy_selection(self, policies, efes, selected_idx, step=None):
        """Log policy selection information to TensorBoard."""
        step = step or self.step_counter
        with self.writer.as_default():
            # Log EFE for top 5 policies
            sorted_indices = np.argsort(efes)
            for i, idx in enumerate(sorted_indices[:5]):
                tf.summary.scalar(f'policy/top{i+1}_efe', efes[idx], step=step)
            
            # Log EFE components for selected policy
            tf.summary.scalar('policy/selected_ambiguity', 
                             policies[selected_idx].ambiguity, step=step)
            tf.summary.scalar('policy/selected_risk', 
                             policies[selected_idx].risk, step=step)
            
            # Log action distribution (for first action in policy)
            actions = [p.actions[0] for p in policies]
            tf.summary.histogram('policy/action_distribution', actions, step=step)
    
    def log_step(self, observation, action, reward, next_observation, 
                belief_state, free_energy, policy_info):
        """Log all information for a single step."""
        self.log_belief_state(belief_state)
        self.log_free_energy(free_energy)
        self.log_policy_selection(**policy_info)
        
        with self.writer.as_default():
            tf.summary.scalar('performance/reward', reward, step=self.step_counter)
            
            # Log observations and actions
            for i, obs in enumerate(observation):
                tf.summary.scalar(f'observation/{i}', obs, step=self.step_counter)
            for i, act in enumerate(action):
                tf.summary.scalar(f'action/{i}', act, step=self.step_counter)
        
        self.step_counter += 1
    
    def log_episode(self, episode_num, total_reward, additional_metrics=None):
        """Log episode-level information."""
        with self.writer.as_default():
            tf.summary.scalar('episode/reward', total_reward, step=episode_num)
            
            if additional_metrics:
                for key, value in additional_metrics.items():
                    tf.summary.scalar(f'episode/{key}', value, step=episode_num)
        
        self.episode_counter += 1
```

### Visualizing Camera Inputs

For visual tasks, it's important to visualize what the fly sees:

```python
def log_camera_images(logger, observation, belief_state, step=None):
    """Log camera images and visual attention maps."""
    step = step or logger.step_counter
    
    # Extract camera images from observation
    left_eye = extract_eye_image(observation, 'left')
    right_eye = extract_eye_image(observation, 'right')
    
    # Generate attention maps from belief state
    left_attention = generate_attention_map(belief_state, 'left')
    right_attention = generate_attention_map(belief_state, 'right')
    
    with logger.writer.as_default():
        # Log raw camera images
        tf.summary.image('vision/left_eye', left_eye, step=step)
        tf.summary.image('vision/right_eye', right_eye, step=step)
        
        # Log attention maps
        tf.summary.image('vision/left_attention', left_attention, step=step)
        tf.summary.image('vision/right_attention', right_attention, step=step)
        
        # Log combined visualization (eye image with attention overlay)
        left_combined = overlay_attention(left_eye, left_attention)
        right_combined = overlay_attention(right_eye, right_attention)
        tf.summary.image('vision/left_combined', left_combined, step=step)
        tf.summary.image('vision/right_combined', right_combined, step=step)
```

### 3D Visualizations

For walking and flying tasks, 3D visualizations help understand the agent's behavior:

```python
def create_3d_trajectory_visualization(beliefs, true_positions, policies):
    """Create a 3D visualization of the agent's trajectory and beliefs."""
    import plotly.graph_objects as go
    
    # Create figure
    fig = go.Figure()
    
    # Plot true trajectory
    fig.add_trace(go.Scatter3d(
        x=true_positions[:, 0],
        y=true_positions[:, 1],
        z=true_positions[:, 2],
        mode='markers+lines',
        name='True Position',
        marker=dict(size=4, color='blue'),
        line=dict(color='blue', width=2)
    ))
    
    # Plot believed trajectory
    fig.add_trace(go.Scatter3d(
        x=beliefs[:, 0],
        y=beliefs[:, 1],
        z=beliefs[:, 2],
        mode='markers+lines',
        name='Believed Position',
        marker=dict(size=4, color='red'),
        line=dict(color='red', width=2)
    ))
    
    # Plot uncertainty as ellipsoids
    for i in range(0, len(beliefs), 10):  # Plot every 10th point for clarity
        add_uncertainty_ellipsoid(fig, beliefs[i], variances[i])
    
    # Plot predicted trajectories for selected policies
    for i, policy in enumerate(selected_policies):
        if i > 5:  # Only show a few for clarity
            break
        predicted_traj = policy.predicted_trajectory
        fig.add_trace(go.Scatter3d(
            x=predicted_traj[:, 0],
            y=predicted_traj[:, 1],
            z=predicted_traj[:, 2],
            mode='lines',
            name=f'Policy {i}',
            line=dict(color=f'rgba(0, 255, 0, {0.8 - i*0.15})', width=1)
        ))
    
    # Set layout
    fig.update_layout(
        title="Agent Trajectory and Beliefs",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position"
        )
    )
    
    return fig
```

## Integration with FlyBody Visualization

The Active Inference visualization tools integrate with FlyBody's existing visualization:

1. **Overlays on simulation**: Adding belief state information to the standard rendering
2. **Side-by-side plots**: Showing internal variables alongside the simulation
3. **Timeline visualization**: Correlating agent decisions with environment events

Example integration:

```python
# In the main training loop
for episode in range(num_episodes):
    # ... training code ...
    
    # Standard FlyBody visualization
    frames = run_episode(env, agent, render=True)
    
    # Active Inference specific visualizations
    belief_frames = visualize_beliefs(agent.logged_beliefs)
    policy_frames = visualize_policies(agent.logged_policies)
    
    # Combine the visualizations
    combined_frames = combine_frames([frames, belief_frames, policy_frames])
    
    # Save to video
    save_video(combined_frames, f"episode_{episode}_visualization.mp4")
```

## Analyzing the Results

The logged data enables various analyses:

1. **Correlation analysis**: How belief accuracy relates to performance
2. **Causal analysis**: How specific observations affect policy selection
3. **Model comparison**: Comparing different Active Inference implementations
4. **Ablation studies**: Understanding the importance of different components

Example analysis script:

```python
def analyze_active_inference_results(log_dir):
    """Analyze results from Active Inference experiments."""
    # Load the data
    data = load_tensorboard_data(log_dir)
    
    # Extract key metrics
    free_energy = data['free_energy/total']
    rewards = data['performance/reward']
    belief_uncertainty = data['belief/total_uncertainty']
    
    # Analysis 1: Correlation between free energy and reward
    correlation = np.corrcoef(free_energy, rewards)[0, 1]
    print(f"Correlation between free energy and reward: {correlation:.4f}")
    
    # Analysis 2: How uncertainty affects policy selection
    uncertainty_vs_exploration = analyze_uncertainty_impact(
        belief_uncertainty, 
        data['policy/action_distribution']
    )
    
    # Analysis 3: Learning progress over time
    learning_curve = compute_smoothed_rewards(rewards, window=100)
    
    # Generate analysis plots
    plot_correlation(free_energy, rewards, "Free Energy vs. Reward")
    plot_uncertainty_impact(uncertainty_vs_exploration)
    plot_learning_curve(learning_curve)
    
    return {
        "correlation": correlation,
        "uncertainty_impact": uncertainty_vs_exploration,
        "learning_curve": learning_curve
    }
```

## Extending the Visualization

The Active Inference visualization framework is designed to be extensible:

1. **Custom metrics**: Add your own metrics to track
2. **Alternative visualizations**: Implement different ways to visualize the same data
3. **Interactive dashboards**: Create Dash or Streamlit dashboards for exploration
4. **Comparative visualization**: Compare multiple agent architectures 