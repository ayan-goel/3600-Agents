# import subprocess

# NUM_GAMES = 20
# AGENT_A = "Durant"
# AGENT_B = "Maldini"

# a_wins = 0
# b_wins = 0
# draws = 0

# for i in range(1, NUM_GAMES + 1):
#     print(f"\n======================")
#     print(f"       GAME {i}")
#     print(f"======================")

#     # Run one game
#     result = subprocess.run(
#         ["python3", "engine/run_local_agents.py", AGENT_A, AGENT_B],
#         capture_output=True,
#         text=True,
#     )

#     # Show the full game log in the terminal
#     print(result.stdout)

#     # Very simple result parsing
#     out = result.stdout
#     if "PLAYER_A wins" in out:
#         a_wins += 1
#         print(f"Result: {AGENT_A} (A) wins")
#     elif "PLAYER_B wins" in out:
#         b_wins += 1
#         print(f"Result: {AGENT_B} (B) wins")
#     else:
#         draws += 1
#         print("Result: draw or unknown (no PLAYER_X line found)")

# print("\n======================")
# print(f"Summary over {NUM_GAMES} games:")
# print(f"{AGENT_A} (A) wins: {a_wins}")
# print(f"{AGENT_B} (B) wins: {b_wins}")
# print(f"Draws / unknown: {draws}")
# print("======================")



import subprocess

NUM_GAMES = 20
AGENT_A = "Durant"
AGENT_B = "Maldini"

a_wins = 0
b_wins = 0
draws = 0

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
        # Optional: very short line so you know what happened
        print(f"Game {i}: {AGENT_A} (A) wins (log omitted)")
    elif "PLAYER_B wins" in out:
        b_wins += 1
        # Print full log only when Durant (A) loses
        print("\n======================")
        print(f"       GAME {i}")
        print("======================")
        print(out)
        print(f"Result: {AGENT_B} (B) wins (Durant loses)")
    else:
        draws += 1
        # You can treat draws like losses if you want logs too
        print("\n======================")
        print(f"       GAME {i}")
        print("======================")
        print(out)
        print("Result: draw or unknown (no PLAYER_X line found)")

print("\n======================")
print(f"Summary over {NUM_GAMES} games:")
print(f"{AGENT_A} (A) wins: {a_wins}")
print(f"{AGENT_B} (B) wins: {b_wins}")
print(f"Draws / unknown: {draws}")
print("======================")
