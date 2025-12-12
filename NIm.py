import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
import pandas as pd
import json
import zipfile
import io
from copy import deepcopy
import math
import time

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Strategic RL Nim",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

st.title("Strategic Nim RL Arena")
st.markdown("""
Watch two Reinforcement Learning agents master the ancient strategy game of **Nim** through intelligent learning.

**🎯 Nim Rules:**
- Start with 4 piles of matchsticks (default: 1, 3, 5, 7)
- Players alternate turns
- On each turn, remove ANY number of sticks from ONE pile
- **Goal:** Force your opponent to take the last stick (last to take LOSES!)

**Core Algorithmic Components:**
- 🎓 **Q-Learning** - Value-based reinforcement learning
- 🌳 **Monte Carlo Tree Search (MCTS)** - Strategic simulation
- 🎯 **Minimax with Alpha-Beta** - Game tree search
- 📊 **Experience Replay** - Enhanced learning
""")

# ============================================================================
# Nim Game Environment
# ============================================================================

class Nim:
    def __init__(self, piles=[1, 3, 5, 7]):
        self.initial_piles = piles[:]
        self.reset()
    
    def reset(self):
        self.piles = self.initial_piles[:]
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        """Return hashable state representation"""
        return tuple(sorted(self.piles))
    
    def get_available_actions(self):
        """Return list of legal moves: (pile_index, num_to_remove)"""
        actions = []
        for i, pile in enumerate(self.piles):
            for take in range(1, pile + 1):
                actions.append((i, take))
        return actions
    
    def make_move(self, action):
        """Execute move and return (state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        pile_idx, take = action
        
        if pile_idx >= len(self.piles) or take > self.piles[pile_idx] or take < 1:
            # Invalid move
            return self.get_state(), -100, True
        
        self.piles[pile_idx] -= take
        self.move_history.append((self.current_player, action))
        
        # Check if game over (all piles empty)
        if sum(self.piles) == 0:
            self.game_over = True
            # Current player took the last stick and LOSES
            self.winner = 1 - self.current_player
            reward = -50  # Penalty for losing
            return self.get_state(), reward, True
        
        # Switch player
        self.current_player = 1 - self.current_player
        return self.get_state(), 0, False
    
    def copy(self):
        """Create a copy of the environment"""
        new_env = Nim(self.initial_piles)
        new_env.piles = self.piles[:]
        new_env.current_player = self.current_player
        new_env.game_over = self.game_over
        new_env.winner = self.winner
        return new_env

# ============================================================================
# MCTS Node
# ============================================================================

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
    
    def uct_value(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, exploration=1.41):
        return max(self.children, key=lambda c: c.uct_value(exploration))
    
    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)

# ============================================================================
# RL Agent with MCTS and Minimax
# ============================================================================

class NimAgent:
    def __init__(self, player_id, lr=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, mcts_sims=50, minimax_depth=3):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.mcts_simulations = mcts_sims
        self.minimax_depth = minimax_depth
        
        self.q_table = {}
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_score = 0
        self.games_played = 0
    
    def get_q_value(self, state, action):
        action_key = str(action)
        return self.q_table.get((state, action_key), 0.0)
    
    def choose_action(self, env, training=True):
        available_actions = env.get_available_actions()
        if not available_actions:
            return None
        
        # Level 1: Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Level 2: Combined strategy - Q-values, MCTS, and Minimax
        best_action = None
        best_score = -float('inf')
        
        for action in available_actions:
            score = 0.0
            
            # Q-Learning component
            q_value = self.get_q_value(env.get_state(), action)
            score += q_value * 0.3
            
            # MCTS component
            if self.mcts_simulations > 0:
                mcts_score = self._evaluate_action_mcts(env, action)
                score += mcts_score * 0.4
            
            # Minimax component
            if self.minimax_depth > 0:
                minimax_score = self._evaluate_action_minimax(env, action)
                score += minimax_score * 0.3
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else available_actions[0]
    
    def _evaluate_action_mcts(self, env, action):
        """Use MCTS to evaluate an action"""
        sim_env = env.copy()
        sim_env.make_move(action)
        
        if sim_env.game_over:
            return 100 if sim_env.winner == self.player_id else -100
        
        root = MCTSNode(sim_env.get_state())
        root.untried_actions = sim_env.get_available_actions()[:]
        
        for _ in range(self.mcts_simulations):
            node = root
            temp_env = sim_env.copy()
            
            # Selection
            while node.untried_actions == [] and node.children:
                node = node.best_child()
                if node.action:
                    temp_env.make_move(node.action)
            
            # Expansion
            if node.untried_actions and not temp_env.game_over:
                next_action = random.choice(node.untried_actions)
                node.untried_actions.remove(next_action)
                temp_env.make_move(next_action)
                child = MCTSNode(temp_env.get_state(), parent=node, action=next_action)
                node.children.append(child)
                node = child
            
            # Simulation
            reward = self._simulate_playout(temp_env)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        return root.value / max(root.visits, 1)
    
    def _evaluate_action_minimax(self, env, action):
        """Use minimax to evaluate an action"""
        sim_env = env.copy()
        sim_env.make_move(action)
        
        if sim_env.game_over:
            return 100 if sim_env.winner == self.player_id else -100
        
        return self._minimax(sim_env, self.minimax_depth - 1, -float('inf'), float('inf'), False)
    
    def _minimax(self, env, depth, alpha, beta, is_maximizing):
        """Minimax with alpha-beta pruning"""
        if env.game_over:
            if env.winner == self.player_id:
                return 100
            else:
                return -100
        
        if depth == 0:
            return 0
        
        actions = env.get_available_actions()
        if not actions:
            return 0
        
        if is_maximizing:
            max_eval = -float('inf')
            for action in actions[:10]:  # Limit branching
                sim_env = env.copy()
                sim_env.make_move(action)
                eval_score = self._minimax(sim_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in actions[:10]:  # Limit branching
                sim_env = env.copy()
                sim_env.make_move(action)
                eval_score = self._minimax(sim_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _simulate_playout(self, env):
        """Random playout for MCTS"""
        sim_env = env.copy()
        steps = 0
        max_steps = 30
        
        while not sim_env.game_over and steps < max_steps:
            actions = sim_env.get_available_actions()
            if not actions:
                break
            action = random.choice(actions)
            sim_env.make_move(action)
            steps += 1
        
        if sim_env.game_over:
            return 50 if sim_env.winner == self.player_id else -50
        return 0
    
    def update_q_value(self, state, action, reward, next_state, next_actions):
        action_key = str(action)
        current_q = self.get_q_value(state, action)
        
        if next_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_actions], default=0)
        else:
            max_next_q = 0
        
        td_error = reward + self.gamma * max_next_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[(state, action_key)] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_score = 0
        self.games_played = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    """Play one complete game"""
    env.reset()
    agents = [agent1, agent2]
    
    max_moves = 50
    move_count = 0
    
    while not env.game_over and move_count < max_moves:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        state = env.get_state()
        action = current_agent.choose_action(env, training)
        
        if action is None:
            break
        
        next_state, reward, done = env.make_move(action)
        
        if training:
            next_actions = env.get_available_actions()
            current_agent.update_q_value(state, action, reward, next_state, next_actions)
        
        move_count += 1
        
        if done:
            if env.winner is not None:
                winner = agents[env.winner]
                loser = agents[1 - env.winner]
                
                winner.wins += 1
                loser.losses += 1
            
            for agent in agents:
                agent.games_played += 1
    
    return env.winner

# ============================================================================
# Visualization
# ============================================================================

def visualize_piles(env, title="Nim Game State"):
    """Create matplotlib figure of matchstick piles"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    max_pile = max(env.piles) if env.piles else 1
    
    for i, pile_size in enumerate(env.piles):
        x_offset = i * 2.5
        
        # Draw pile label
        ax.text(x_offset + 0.5, -0.5, f"Pile {i+1}", ha='center', fontsize=14, fontweight='bold')
        
        # Draw matchsticks
        for j in range(pile_size):
            # Matchstick body
            stick = Rectangle((x_offset + 0.3, j * 0.3), 0.4, 0.25, 
                            facecolor='#d35400', edgecolor='#8b4513', linewidth=2)
            ax.add_patch(stick)
            
            # Matchstick head
            head = Circle((x_offset + 0.5, j * 0.3 + 0.35), 0.08, 
                         facecolor='#e74c3c', edgecolor='#c0392b', linewidth=1.5)
            ax.add_patch(head)
    
    ax.set_xlim(-0.5, len(env.piles) * 2.5)
    ax.set_ylim(-1, max_pile * 0.3 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Show current state
    state_text = f"State: {env.piles} | Total: {sum(env.piles)} sticks"
    ax.text(len(env.piles) * 1.25, -0.8, state_text, ha='center', fontsize=12)
    
    return fig

# ============================================================================
# Save/Load Functions
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def serialize_q_table(q_table):
    serialized_q = {}
    for (state, action_key), value in q_table.items():
        state_str = json.dumps(list(state))
        key_str = f"{state_str}|{action_key}"
        serialized_q[key_str] = float(value)
    return serialized_q

def deserialize_q_table(serialized_q):
    deserialized_q = {}
    for k_str, value in serialized_q.items():
        try:
            state_str, action_key = k_str.split('|', 1)
            state_list = json.loads(state_str)
            
            # FIX: Convert inner lists back to tuples so they are hashable
            state_data = []
            for item in state_list:
                if isinstance(item, list):
                    state_data.append(tuple(item))
                else:
                    state_data.append(item)
            
            state_tuple = tuple(state_data)
            deserialized_q[(state_tuple, action_key)] = value
        except Exception as e:
            continue
    return deserialized_q

def create_agents_zip(agent1, agent2, config):
    agent1_data = {
        "q_table": serialize_q_table(agent1.q_table),
        "epsilon": float(agent1.epsilon),
        "wins": int(agent1.wins),
        "losses": int(agent1.losses),
        "total_score": int(agent1.total_score),
        "games_played": int(agent1.games_played),
        "mcts_simulations": int(agent1.mcts_simulations),
        "minimax_depth": int(agent1.minimax_depth)
    }
    
    agent2_data = {
        "q_table": serialize_q_table(agent2.q_table),
        "epsilon": float(agent2.epsilon),
        "wins": int(agent2.wins),
        "losses": int(agent2.losses),
        "total_score": int(agent2.total_score),
        "games_played": int(agent2.games_played),
        "mcts_simulations": int(agent2.mcts_simulations),
        "minimax_depth": int(agent2.minimax_depth)
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(agent1_data, cls=NumpyEncoder, indent=2))
        zf.writestr("agent2.json", json.dumps(agent2_data, cls=NumpyEncoder, indent=2))
        zf.writestr("config.json", json.dumps(config, cls=NumpyEncoder, indent=2))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_state = json.loads(zf.read("agent1.json"))
            agent2_state = json.loads(zf.read("agent2.json"))
            config = json.loads(zf.read("config.json"))
            
            agent1 = NimAgent(0, config.get('lr1', 0.1), config.get('gamma1', 0.95),
                            mcts_sims=agent1_state.get('mcts_simulations', 50),
                            minimax_depth=agent1_state.get('minimax_depth', 3))
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state.get('epsilon', 0.0)
            agent1.wins = agent1_state.get('wins', 0)
            agent1.losses = agent1_state.get('losses', 0)
            agent1.total_score = agent1_state.get('total_score', 0)
            agent1.games_played = agent1_state.get('games_played', 0)
            
            agent2 = NimAgent(1, config.get('lr2', 0.1), config.get('gamma2', 0.95),
                            mcts_sims=agent2_state.get('mcts_simulations', 50),
                            minimax_depth=agent2_state.get('minimax_depth', 3))
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.epsilon = agent2_state.get('epsilon', 0.0)
            agent2.wins = agent2_state.get('wins', 0)
            agent2.losses = agent2_state.get('losses', 0)
            agent2.total_score = agent2_state.get('total_score', 0)
            agent2.games_played = agent2_state.get('games_played', 0)
            
            return agent1, agent2, config
    except Exception as e:
        st.error(f"Error loading brain file: {str(e)}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Controls")

# Game Setup
with st.sidebar.expander("🎮 Game Setup", expanded=True):
    pile1 = st.number_input("Pile 1", 1, 20, 1, 1)
    pile2 = st.number_input("Pile 2", 1, 20, 3, 1)
    pile3 = st.number_input("Pile 3", 1, 20, 5, 1)
    pile4 = st.number_input("Pile 4", 1, 20, 7, 1)

with st.sidebar.expander("1. Agent 1 Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.01, 0.5, 0.1, 0.01)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.995, 0.0001, format="%.4f")
    mcts_sims1 = st.slider("MCTS Simulations₁", 0, 500, 200, 10)
    minimax_depth1 = st.slider("Minimax Depth₁", 0, 20, 10, 1)

with st.sidebar.expander("2. Agent 2 Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.01, 0.5, 0.1, 0.01)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.995, 0.0001, format="%.4f")
    mcts_sims2 = st.slider("MCTS Simulations₂", 0, 500, 200, 10)
    minimax_depth2 = st.slider("Minimax Depth₂", 0, 20, 10, 1)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 10000, 500, 10)
    update_freq = st.number_input("Update Every N Games", 5, 100, 50, 5)

with st.sidebar.expander("4. Brain Storage 💾", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        brain_size = len(st.session_state.agent1.q_table) + len(st.session_state.agent2.q_table)
        
        if brain_size > 0:
            st.success(f" Brain Scan: {brain_size} memories found.")
            
            config = {
                "lr1": lr1, "gamma1": gamma1, "epsilon_decay1": epsilon_decay1,
                "lr2": lr2, "gamma2": gamma2, "epsilon_decay2": epsilon_decay2,
                "training_history": st.session_state.get('training_history', None),
                "piles": [pile1, pile2, pile3, pile4]
            }
            
            zip_buffer = create_agents_zip(st.session_state.agent1, st.session_state.agent2, config)
            
            st.download_button(
                label="💾 Download Trained Brains",
                data=zip_buffer,
                file_name="nim_brains.zip",
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Brains are empty! Train the agents first.")
    else:
        st.warning("Initialize agents first.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Brain Snapshot (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("📂 Load Session", use_container_width=True):
            with st.spinner("Restoring neural pathways..."):
                a1, a2, cfg = load_agents_from_zip(uploaded_file)
                if a1:
                    st.session_state.agent1 = a1
                    st.session_state.agent2 = a2
                    
                    if cfg.get("training_history"):
                        st.session_state.training_history = cfg["training_history"]
                    
                    st.toast("Brains Restored Successfully!", icon="♾️")
                    time.sleep(1)
                    st.rerun()

train_button = st.sidebar.button(" Begin Training", use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear All & Reset", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.toast("Arena Reset!", icon="🧹")
    st.rerun()

# ============================================================================
# Initialize Environment and Agents
# ============================================================================

if 'env' not in st.session_state:
    st.session_state.env = Nim([pile1, pile2, pile3, pile4])

if 'agent1' not in st.session_state:
    st.session_state.agent1 = NimAgent(0, lr1, gamma1, epsilon_decay=epsilon_decay1, 
                                      mcts_sims=mcts_sims1, minimax_depth=minimax_depth1)
    st.session_state.agent2 = NimAgent(1, lr2, gamma2, epsilon_decay=epsilon_decay2,
                                      mcts_sims=mcts_sims2, minimax_depth=minimax_depth2)

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
env = st.session_state.env

# Update parameters
agent1.mcts_simulations = mcts_sims1
agent1.minimax_depth = minimax_depth1
agent2.mcts_simulations = mcts_sims2
agent2.minimax_depth = minimax_depth2

# ============================================================================
# Display Stats
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(" Agent 1", f"Q-States: {len(agent1.q_table)}", f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins)
    st.caption(f"MCTS: {agent1.mcts_simulations} | Minimax: {agent1.minimax_depth}")

with col2:
    st.metric(" Agent 2", f"Q-States: {len(agent2.q_table)}", f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins)
    st.caption(f"MCTS: {agent2.mcts_simulations} | Minimax: {agent2.minimax_depth}")

with col3:
    total_games = agent1.games_played
    st.metric("Total Games", total_games)
    if total_games > 0:
        win_rate1 = (agent1.wins / total_games) * 100
        st.caption(f"Agent 1 Win Rate: {win_rate1:.1f}%")

st.markdown("---")

# ============================================================================
# Training Section
# ============================================================================

if train_button:
    st.subheader("🎯 Training in Progress...")
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [],
        'agent2_wins': [],
        'agent1_epsilon': [],
        'agent2_epsilon': [],
        'agent1_q_size': [],
        'agent2_q_size': [],
        'episode': []
    }
    
    for episode in range(1, episodes + 1):
        play_game(env, agent1, agent2, training=True)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_q_size'].append(len(agent1.q_table))
            history['agent2_q_size'].append(len(agent2.q_table))
            history['episode'].append(episode)
            
            progress = episode / episodes
            progress_bar.progress(progress)
            
            win_rate1 = (agent1.wins / episode) * 100
            win_rate2 = (agent2.wins / episode) * 100
            
            status_table = f"""
            | Metric | Agent 1 | Agent 2 |
            |:-------|:-------:|:-------:|
            | **Wins** | {agent1.wins} ({win_rate1:.1f}%) | {agent2.wins} ({win_rate2:.1f}%) |
            | **Epsilon** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | **Q-States** | {len(agent1.q_table):,} | {len(agent2.q_table):,} |
            
            ---
            **Game {episode}/{episodes}** ({progress*100:.1f}%)
            """
            status_container.markdown(status_table)
    
    progress_bar.progress(1.0)
    
    st.session_state.training_history = history
    st.session_state.agent1 = agent1
    st.session_state.agent2 = agent2
    
    st.toast("Training Complete!", icon="🎉")
    time.sleep(1)
    st.rerun()

# ============================================================================
# Display Training Charts
# ============================================================================

if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📈 Training Performance")
    history = st.session_state.training_history
    df = pd.DataFrame(history)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("#### Win Count Over Time")
        chart_data = df[['episode', 'agent1_wins', 'agent2_wins']].set_index('episode')
        st.line_chart(chart_data)
    
    with chart_col2:
        st.write("#### Epsilon Decay")
        epsilon_data = df[['episode', 'agent1_epsilon', 'agent2_epsilon']].set_index('episode')
        st.line_chart(epsilon_data)
    
    st.write("#### Q-Table Growth")
    q_chart = df[['episode', 'agent1_q_size', 'agent2_q_size']].set_index('episode')
    st.line_chart(q_chart)

# ============================================================================
# AI vs AI Battle Arena
# ============================================================================

st.markdown("---")
st.subheader("⚔️ AI vs AI Battle Arena")

battle_col1, battle_col2 = st.columns(2)

with battle_col1:
    if st.button("🎮 Watch Agents Battle!", use_container_width=True, type="primary"):
        st.session_state.battle_active = True
        st.session_state.battle_env = Nim([pile1, pile2, pile3, pile4])
        st.session_state.battle_move = 0
        st.rerun()

with battle_col2:
    if st.button("⏹️ Stop Battle", use_container_width=True):
        st.session_state.battle_active = False
        st.rerun()

if st.session_state.get('battle_active', False):
    battle_env = st.session_state.battle_env
    
    if not battle_env.game_over:
        # Display current state
        current_player_name = f"Agent {battle_env.current_player + 1}"
        fig = visualize_piles(battle_env, f"{current_player_name}'s Turn | Move {st.session_state.battle_move}")
        st.pyplot(fig)
        plt.close(fig)
        
        # Show scores
        st.info(f"**Current Player:** {current_player_name}")
        
        # Make move automatically
        time.sleep(2)
        current_agent = agent1 if battle_env.current_player == 0 else agent2
        action = current_agent.choose_action(battle_env, training=False)
        
        if action:
            pile_idx, num_take = action
            st.success(f"**{current_player_name}** removes **{num_take}** stick(s) from **Pile {pile_idx + 1}**")
            battle_env.make_move(action)
            st.session_state.battle_move += 1
        
        st.rerun()
    else:
        # Game over
        fig = visualize_piles(battle_env, "Battle Complete!")
        st.pyplot(fig)
        plt.close(fig)
        
        if battle_env.winner == 0:
            st.success("🏆 **Agent 1 Wins!**")
        else:
            st.error("🏆 **Agent 2 Wins!**")
        
        st.session_state.battle_active = False

# ============================================================================
# Human vs AI Mode
# ============================================================================

st.markdown("---")
st.subheader("👤 Human vs AI")

if 'human_game' not in st.session_state:
    st.session_state.human_game = None
    st.session_state.human_is_player = 0

col_h1, col_h2 = st.columns(2)

with col_h1:
    if st.button("🎮 Start Game (You Go First)", use_container_width=True):
        st.session_state.human_game = Nim([pile1, pile2, pile3, pile4])
        st.session_state.human_is_player = 0
        st.rerun()

with col_h2:
    if st.button(" Start Game (AI Goes First)", use_container_width=True):
        st.session_state.human_game = Nim([pile1, pile2, pile3, pile4])
        st.session_state.human_is_player = 1
        # AI makes first move
        if agent1.q_table:
            action = agent1.choose_action(st.session_state.human_game, training=False)
            if action:
                st.session_state.human_game.make_move(action)
        st.rerun()

if st.session_state.human_game and not st.session_state.human_game.game_over:
    game = st.session_state.human_game
    
    fig = visualize_piles(game, "Current Game State")
    st.pyplot(fig)
    plt.close(fig)
    
    if game.current_player == st.session_state.human_is_player:
        st.info("🎮 Your turn! Select a pile and number of sticks to remove.")
        
        hcol1, hcol2, hcol3 = st.columns(3)
        
        with hcol1:
            pile_choice = st.selectbox("Select Pile", range(len(game.piles)), 
                                      format_func=lambda x: f"Pile {x+1} ({game.piles[x]} sticks)")
        
        with hcol2:
            # Get actual sticks, but ensure max is at least 1 to prevent UI crash
            pile_sticks = game.piles[pile_choice] if pile_choice < len(game.piles) else 0
            safe_max = max(1, pile_sticks)
            
            num_take = st.number_input("Remove How Many?", 1, safe_max, 1)
        
        with hcol3:
            st.write("")
            st.write("")
            if st.button("✅ Make Move", use_container_width=True):
                game.make_move((pile_choice, num_take))
                
                # AI's turn
                if not game.game_over and agent1.q_table:
                    time.sleep(0.5)
                    action = agent1.choose_action(game, training=False)
                    if action:
                        game.make_move(action)
                
                st.rerun()
    else:
        st.warning(" AI is thinking...")
        time.sleep(1)
        st.rerun()

elif st.session_state.human_game and st.session_state.human_game.game_over:
    game = st.session_state.human_game
    
    fig = visualize_piles(game, "Game Over!")
    st.pyplot(fig)
    plt.close(fig)
    
    if game.winner == st.session_state.human_is_player:
        st.success("🎉 Congratulations! You won!")
    else:
        st.error(" AI wins! Better luck next time!")
    
    if st.button("🔄 Play Again", use_container_width=True):
        st.session_state.human_game = None
        st.rerun()

st.markdown("---")
st.caption("Strategic RL Nim Arena | Ancient Strategy Meets Modern AI ")
