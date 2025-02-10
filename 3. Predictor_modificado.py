import argparse
import os
import pathlib
import numpy as np
import torch
from stable_baselines3 import PPO
from gym_locm import agents
from gym_locm.envs import LOCMBattleEnv
from scipy.special import softmax

base_path = str(pathlib.Path(__file__).parent.absolute())
print(base_path)
def get_arg_parser():
    p = argparse.ArgumentParser(
        description="This is a predictor for trained RL drafts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--draft-1", help="Path to first draft model", required=True)
    p.add_argument("--draft-2", help="Path to second draft model", required=True)
    p.add_argument("--approach-1", help="Approach for first model (immediate, history, lstm)", required=True)
    p.add_argument("--approach-2", help="Approach for second model (immediate, history, lstm)", required=True)

    return p

def run_battle(deck1, deck2):
    # Simulates the battle using the MaxAttackBattleAgent within the same thread
    print("[DEBUG] Running battle simulation...")
    env = LOCMBattleEnv()
    env.state.players[0].deck = deck1  # Assign deck1 to player 1
    env.state.players[1].deck = deck2  # Assign deck2 to player 2
    battle_agent_1 = agents.GreedyBattleAgent()
    battle_agent_2 = agents.GreedyBattleAgent()
    
    obs = env.reset()
    done = False
    while not done:
        if env.state.current_player == 0:
            action = battle_agent_1.act(env.state)
        else:
            action = battle_agent_2.act(env.state)
        
        obs, reward, done, info = env.step(action)
    
    #print("[DEBUG] Battle simulation finished. Winner:", "Player 1 (No Optimizado)" if reward > 0 else "Player 2 (Optimizado)")
    return reward  # Returns the result of the match

def load_model(model_path):
    print(f"[DEBUG] Loading RL model from {model_path}")
    model = PPO.load(model_path)
    print("[DEBUG] RL model loaded successfully.")
    return model

def draft_deck(model, approach):
    deck = []
    for i in range(30):  # Simulate 30 rounds of drafting
        obs = np.random.rand(1, 48)  # Default observation shape

        # Si el modelo es "history", usa 528 dimensiones
        if approach == "history":
            obs = np.random.rand(528)

        action, _ = model.predict(obs)
        deck.append(f"Picked Card {action}")
        print(f"[DEBUG] Turn {i+1}: Model chose {action}")
    return deck

def main():
    # Main function to orchestrate the draft and battle evaluation
    args = get_arg_parser().parse_args()
    print("[DEBUG] Predictor started with the following arguments:")
    print(args)
    
    # Load the RL draft models
    model_1 = load_model(base_path + "/" + args.draft_1)
    model_2 = load_model(base_path + "/" + args.draft_2)
    
    # Generate decks using the RL models
    deck1 = draft_deck(model_1, args.approach_1)
    deck2 = draft_deck(model_2, args.approach_2)
    
    # Compare shared cards between both decks
    shared_cards = set(deck1).intersection(set(deck2))
    print(f"[DEBUG] Shared cards count: {len(shared_cards)}")

    
    # Run the battle between the two decks
    num_games = 1000
    wins_model_1 = 0
    wins_model_2 = 0
    
    for _ in range(num_games):
        result = run_battle(deck1, deck2)
        if result > 0:
            wins_model_1 += 1
        else:
            wins_model_2 += 1
    
    print(f"[DEBUG] Games Played: {num_games}")
    print(f"[DEBUG] Model 1 ({args.approach_1}) Wins: {wins_model_1} ({(wins_model_1/num_games)*100:.2f}%)")
    print(f"[DEBUG] Model 2 ({args.approach_2}) Wins: {wins_model_2} ({(wins_model_2/num_games)*100:.2f}%)")
    
    print("[DEBUG] Predictor finished execution. Battle Result: Model 1 Wins:", wins_model_1, "Model 2 Wins:", wins_model_2)
    

if __name__ == "__main__":
    main()
