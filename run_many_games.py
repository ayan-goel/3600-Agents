import subprocess

NUM_GAMES = 30
AGENT_A = "Durant"
AGENT_B = "MaldiniCopy"

a_wins = 0
b_wins = 0
draws = 0

def print_last_turn(game_index: int, out: str, result_label: str) -> None:
    """Print only the last turn (or last few lines if format is unknown)."""
    lines = out.splitlines()
    if not lines:
        print(f"\n======================")
        print(f"       GAME {game_index}")
        print(f"======================")
        print("[No output captured]")
        print(result_label)
        return

    # Try to find the last line that looks like a turn header
    last_turn_idx = -1
    for i, line in enumerate(lines):
        if "Turn" in line or "TURN" in line:
            last_turn_idx = i

    if last_turn_idx == -1:
        # Fallback: just show the last 10 lines
        snippet = lines[-10:]
    else:
        # Show from the last turn header to the end
        snippet = lines[last_turn_idx:]

    print("\n======================")
    print(f"       GAME {game_index}")
    print("======================")
    print("\n".join(snippet))
    print(result_label)


for i in range(1, NUM_GAMES + 1):
    # Run one game
    result = subprocess.run(
        ["python3", "engine/run_local_agents.py", AGENT_A, AGENT_B],
        capture_output=True,
        text=True,
    )

    out = result.stdout

    # Very simple result parsing
    if "PLAYER_A wins" in out:
        a_wins += 1
        result_label = f"Result: {AGENT_A} (A) wins"
    elif "PLAYER_B wins" in out:
        b_wins += 1
        result_label = f"Result: {AGENT_B} (B) wins"
    else:
        draws += 1
        result_label = "Result: draw or unknown (no PLAYER_X line found)"

    # For EVERY game, show only the last turn snippet + result
    print_last_turn(i, out, result_label)

print("\n======================")
print(f"Summary over {NUM_GAMES} games:")
print(f"{AGENT_A} (A) wins: {a_wins}")
print(f"{AGENT_B} (B) wins: {b_wins}")
print(f"Draws / unknown: {draws}")
print("======================")
