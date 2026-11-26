# Goore Game

An implementation of the **Goore Game** with learning automata, for teaching and self-study. It is used
as course material by Dr. Ali Mohammad Saghiri ([homepage](https://amsaghiri.github.io/)).

## The game

A group of players votes *yes* or *no* in each round. A referee looks only at the **fraction of yes
votes** and then rewards each player independently with a probability that depends on that fraction.
The players cannot see each other, cannot communicate, and do not know the referee's rule. Each player
only learns whether it was rewarded.

The surprising result is that if every player is a simple learning automaton, the group still finds the
fraction of yes votes that the referee rewards most.

In this implementation the reward probability is

```
G(x) = 0.9 · exp( −(0.7 − x)² / 0.0625 )
```

where *x* is the fraction of yes votes, so the best outcome is 70% yes.

## What is in the repository

| File | What it is |
| --- | --- |
| `Goor_Game.py` | The complete game as a script: 10 players, each a two-action learning automaton. |
| `Goore_Game.ipynb` | Notebook version with a fixed learning rate. It also plots how many players vote yes over time. |
| `Goore_Game_adaptive_rate.ipynb` | The same game with an adaptive learning rate: `lambda_1` grows linearly from 0.001 to 0.1 during the run, and the plot marks the optimal number of yes votes. |

The notebooks open directly in Google Colab from the badge at the top of each one.

## Run it

Requires Python 3 with NumPy and Matplotlib.

```bash
pip install numpy matplotlib
python Goor_Game.py
```

The script prints the final action probabilities of every player and saves two figures per player:
`favor_action<N>.png` (the action the player prefers over time) and `entropy<N>.png` (how undecided
the player is over time).

In a typical run, 7 of the 10 players end up voting yes and 3 voting no, which is exactly the 70% the
referee rewards most. Which players end up on which side differs from run to run.

## How it works

- `Automata` – a two-action learning automaton with a linear reward–inaction update: when a player is
  rewarded, the probability of the action it just took increases by `lambda_1 · (1 − p)`; a penalty
  changes nothing.
- `Environment` – the referee. `get_reward(rate_yes)` returns 1 with probability *G(rate_yes)*.
- `Node` and `Graph` – containers for the players.

## Things to try

| Change | Where | Question |
| --- | --- | --- |
| Move the peak from 0.7 to 0.3 | `Environment.get_reward` | How many players vote yes now? |
| `number_players` from 10 to 20 | settings | Is the outcome closer to the peak? Why? |
| `lambda_1` from 0.01 to 0.1 | settings | Faster, but is the final split still right? |
| Starting probabilities | `probability_players` | Does the starting point change who votes yes? |

## References

The Goore Game was introduced by Tsetlin and is analysed in detail by Narendra and Thathachar.

- M. L. Tsetlin, *Automaton Theory and Modeling of Biological Systems*. New York: Academic Press, 1973.
- K. S. Narendra and M. A. L. Thathachar, *Learning Automata: An Introduction*. Englewood Cliffs, NJ: Prentice Hall, 1989.

## Related repositories

- [Learning-Automata](https://github.com/amsaghiri/Learning-Automata) – a single automaton, the best place to start
- [Multi-armed-bandit](https://github.com/amsaghiri/Multi-armed-bandit) – the same game played by ε-greedy bandit agents
