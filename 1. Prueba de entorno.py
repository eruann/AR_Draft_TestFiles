import gym
import gym_locm
import numpy as np

# Crear el entorno de LOCM
env = gym.make("LOCM-draft-v0")

# Definir estrategias heurísticas: Random y Max-Attack
def random_policy(state):
    """Elige una acción aleatoria en cada turno."""
    return env.action_space.sample()

def max_attack_policy(state):
    """Elige la carta con mayor ataque disponible."""
    num_cards = 3  # Siempre hay 3 opciones en el draft
    card_features = 16  # Cada carta tiene 16 atributos
    cards = state.reshape(num_cards, card_features)
    attack_values = cards[:, 1]  # El índice 1 corresponde al ataque
    return np.argmax(attack_values)

# Evaluar cada estrategia con 1,000 partidas separando Jugador 1 y 2
def evaluate_policy(policy, num_games=1000):
    """Evalúa una política jugando varias partidas y separando Jugador 1 y 2."""
    wins_p1 = 0
    wins_p2 = 0

    for i in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)

        if reward > 0:
            if i % 2 == 0:
                wins_p1 += 1
            else:
                wins_p2 += 1

    return (wins_p1 / (num_games // 2)) * 100, (wins_p2 / (num_games // 2)) * 100

# Ejecutar las evaluaciones
print("Evaluando agentes heurísticos...")
random_win_rate_p1, random_win_rate_p2 = evaluate_policy(random_policy)
max_attack_win_rate_p1, max_attack_win_rate_p2 = evaluate_policy(max_attack_policy)

# Mostrar los resultados
print(f"Tasa de victoria de Random (Jugador 1): {random_win_rate_p1:.2f}%")
print(f"Tasa de victoria de Random (Jugador 2): {random_win_rate_p2:.2f}%")
print(f"Tasa de victoria de Max-Attack (Jugador 1): {max_attack_win_rate_p1:.2f}%")
print(f"Tasa de victoria de Max-Attack (Jugador 2): {max_attack_win_rate_p2:.2f}%")
