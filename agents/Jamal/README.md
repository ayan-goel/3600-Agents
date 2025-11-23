Jamal - Hybrid Heuristic + Neural Agent
=======================================

Components
- Agent (`agents/Jamal/agent.py`): CNN policy (from Dontique) + light heuristics reweighting.
- Offline training (`agents/Jamal/train_jamal.py`): Supervised on `agents/matches/*.json`.
- Data loader (`agents/Jamal/data.py`): Parses match JSON and reconstructs approximate states.
- Slurm script (`agents/Jamal/train_jamal.sh`): Run on PACE.

Quick start (local)
1) Train from existing matches:
   - `python -m agents.Jamal.train_jamal --match_dir agents/matches --epochs 8 --limit 3000`
   - Outputs `agents/Jamal/jamal_weights.pt`.
2) Test Jamal:
   - `PYTHONPATH=$PWD python engine/run_local_agents.py Jamal Fluffy`
   - `PYTHONPATH=$PWD python engine/run_local_agents.py Jamal Messi`

3) Fine-tune with self-play + opponent pool (eval-gated):
   - `python -m agents.Jamal.selfplay --iterations 12 --games-per-opp 12`
   - Saves improved weights to `agents/Jamal/jamal_weights.pt` only if win rate vs pool (Fluffy, Messi) improves.

PACE (Slurm)
- `sbatch agents/Jamal/train_jamal.sh`

Notes
- State reconstruction from JSON approximates eggs/turds and uses left_behind to mark eggs. This is sufficient for supervised warm-start, then you can add self-play (extend with Dontique’s pipeline) for improvement.
- The agent runs even without weights (falls back to heuristics+random priors), but you should train first.


