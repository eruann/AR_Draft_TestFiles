import gym
import gym_locm
from gym_locm.engine.card import Creature
from gym_locm.util import encode_card, encode_state_draft
import numpy as np

# Crear una carta de prueba para verificar encode_card()
test_card = Creature(
    card_id=1,
    name="Test Creature",
    card_type=0,  # Criatura
    cost=5,
    attack=7,
    defense=3,
    keywords="BC",  # Breakthrough, Charge
    player_hp=0,
    enemy_hp=0,
    card_draw=1,
    area=0,
    text="",
    instance_id=1
)

encoded_card = encode_card(test_card)
print("Representación de la carta con encode_card():", encoded_card)
print("Dimensión esperada:", len(encoded_card), "(Debe ser 16)")

# Crear un estado de draft ficticio con 3 cartas disponibles
class MockState:
    def __init__(self, cards):
        self.k = len(cards)  # Número de cartas en el draft
        self.current_player = self
        self.hand = cards

    def is_draft(self):
        return True

# Crear 3 cartas distintas para la prueba de encode_state_draft()
cards = [
    Creature(2, "Card A", 0, 3, 4, 5, "D", 0, 0, 0, 0, "", 2),
    Creature(3, "Card B", 0, 2, 2, 8, "L", 0, 0, 1, 0, "", 3),
    Creature(4, "Card C", 0, 6, 6, 2, "G", 0, 0, 0, 0, "", 4)
]

mock_state = MockState(cards)

# Aplicar encode_state_draft()
encoded_state = encode_state_draft(mock_state)

print("Representación del estado del draft con encode_state_draft():", encoded_state)
print("Dimensión esperada:", encoded_state.shape, "(Debe ser (48,))")

# Verificar normalización
print("Valores máximos y mínimos en el estado:", np.max(encoded_state), np.min(encoded_state))
