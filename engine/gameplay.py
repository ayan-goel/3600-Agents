import time
from collections.abc import Iterable
from typing import Set, Tuple

from board_utils import get_board_string
from game.board import Board
from game.enums import Direction, MoveType, Result, ResultArbiter, WinReason
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager
from player_process import PlayerProcess


def init_display(board, player_a_name, player_b_name):
    # print(player_a_name+ " vs. "+player_b_name)
    # print("\nA to play" if board.is_as_turn() else "\nB to play")
    pass


# prints board to terminal, clearing on each round
def print_board(board: Board, trapdoors: Set[Tuple[int, int]], clear_screen):
    player_map, a_eggs, b_eggs, a_turds, b_turds = get_board_string(board, trapdoors)

    import os

    if clear_screen:
        if os.name == "nt":
            os.system("cls || clear")
        else:
            os.system("clear || cls")

    board_list = []

    board_list.append("\n--- TURN " + str(board.turn_count) + ": ")
    if board.is_as_turn:
        board_list.append(f"A to play, Time left:{board.player_time:.2f}\n")
    else:
        board_list.append(f"B to play, Time left:{board.player_time:.2f}\n")

    board_list.append(player_map)
    board_list.append(f" EGGS  A:{a_eggs: <2d} B:{b_eggs: <2d}\n")
    board_list.append(f" TURDS A:{a_turds: <2d} B:{b_turds: <2d}\n")

    print("".join(board_list), end="")


# prints a player's move on a given turn
def print_moves(player_as_turn, moves, timer):
    dir, movetype = moves
    try:
        if player_as_turn:
            print("A plays:", end="")
        else:
            print("B plays:", end="")
        if moves is None:
            print("None", end="")
        else:
            print(f"({Direction(dir).name}, {MoveType(movetype).name})", end="")
    except:
        print("Invalid", end="")

    print(f" in {timer:.3f} seconds")


def validate_submission(
    directory_a, player_a_name, limit_resources=False, use_gpu=False
):
    import importlib.util
    import subprocess
    import sys
    import traceback
    from multiprocessing import Process, Queue, set_start_method

    player_a_process = None

    try:
        if not directory_a in sys.path:
            sys.path.append(directory_a)

        play_time = 360
        extra_ret_time = 5

        # setup main thread queue for getting results
        main_q = Queue()

        # setup two thread queues for passing commands to players
        player_a_q = Queue()
        map_to_play = GameMap()
        trapdoor_manager = TrapdoorManager(map_to_play)
        game_board = Board(map_to_play, play_time, build_history=False)
        spawns = trapdoor_manager.choose_spawns()
        trapdoor_manager.choose_trapdoors()
        game_board.chicken_player.start(spawns[0], 0)
        game_board.chicken_enemy.start(spawns[1], 1)

        queues = [player_a_q, main_q]

        out_queue = Queue()

        player_a_process = PlayerProcess(
            True,
            player_a_name,
            directory_a,
            player_a_q,
            main_q,
            limit_resources,
            use_gpu,
            out_queue,
            user_name="player_a_user",
            group_name="player_a",
        )
        player_a_process.start()

        ok = main_q.get(block=True, timeout=10)
        message = ""

        if not ok:
            message = "Failed to initialize agent"
        else:
            init_timeout = 10
            ok, message = player_a_process.run_timed_constructor(
                game_board, init_timeout, 10
            )

        if ok:
            samples = trapdoor_manager.sample_trapdoors(spawns[0])
            moves, timer, message = player_a_process.run_timed_play(
                game_board, samples, game_board.player_time, extra_ret_time
            )
            ok = not moves is None

        terminate_validation(player_a_process, queues, out_queue)
        return ok, message
    except:
        print(traceback.format_exc())

        if player_a_process.process:
            terminate_validation(player_a_process, queues, out_queue)
        return False, traceback.format_exc()


def delete_module(name):
    import sys

    if name in sys.modules:
        del sys.modules[name]


def terminate_validation(process_a, queues, out_queue):
    delete_module("player_a.agent")
    delete_module("player_a")

    process_a.terminate_process_and_children()

    for q in queues:
        try:
            while True:
                q.get_nowait()
        except:
            pass

    try:
        while True:
            out_queue.get_nowait()
    except:
        pass


# Listener function to continuously listen to the queue
def listen_for_output(output_queue, stop_event):
    while not stop_event.is_set():
        try:
            print(output_queue.get(timeout=1))  # Wait for 1 second for output
        except:
            continue  # No output yet, continue listening


def play_game(
    directory_a,
    directory_b,
    player_a_name,
    player_b_name,
    display_game=False,
    delay=0,
    clear_screen=True,
    record=True,
    limit_resources=False,
    use_gpu=False,
):
    # setup main environment, import player modules
    import os
    import sys
    import threading
    import traceback
    from multiprocessing import Process, Queue

    if not directory_a in sys.path:
        sys.path.append(directory_a)

    if not directory_b in sys.path:
        sys.path.append(directory_b)

    play_time = 360
    extra_ret_time = 5
    init_timeout = 10

    if not limit_resources:
        init_timeout = 20
        play_time = 360

    # setup main thread queue for getting results
    main_q_a = Queue()
    main_q_b = Queue()

    # setup two thread queues for passing commands to players
    player_a_q = Queue()
    player_b_q = Queue()

    # game init
    map_to_play = GameMap()
    trapdoor_manager = TrapdoorManager(map_to_play)
    game_board = Board(map_to_play, play_time, build_history=record)
    spawns = trapdoor_manager.choose_spawns()
    trapdoor_locations = trapdoor_manager.choose_trapdoors()
    print(f"Trapdoors: {trapdoor_locations}")
    game_board.chicken_player.start(spawns[0], 0)
    game_board.chicken_enemy.start(spawns[1], 1)

    out_queue = Queue()
    stop_event = None
    if not limit_resources:
        stop_event = threading.Event()
        listener_thread = threading.Thread(
            target=listen_for_output, args=(out_queue, stop_event)
        )
        listener_thread.daemon = True
        listener_thread.start()

    queues = [player_a_q, player_b_q, main_q_a, main_q_b]

    # startup two player processes
    player_a_process = PlayerProcess(
        True,
        player_a_name,
        directory_a,
        player_a_q,
        main_q_a,
        limit_resources,
        use_gpu,
        out_queue,
        user_name="player_a_user",
        group_name="player_a",
    )

    player_b_process = PlayerProcess(
        False,
        player_b_name,
        directory_b,
        player_b_q,
        main_q_b,
        limit_resources,
        use_gpu,
        out_queue,
        user_name="player_b_user",
        group_name="player_b",
    )

    success_a = False
    success_b = False

    message_a = ""
    message_b = ""

    

    try:
        player_a_process.start()
        success_a = main_q_a.get(block=True, timeout=10)
        player_a_process.pause_process_and_children()
    except Exception as e:
        message_a = traceback.format_exc()
        print(f"Player a crashed during initialization: {message_a}")

    try:
        player_b_process.start()
        success_b = main_q_b.get(block=True, timeout=10)
        player_b_process.pause_process_and_children()
    except Exception as e:
        message_b = traceback.format_exc()
        print(f"Player b crashed during initialization: {message_b}")

    if success_a and success_b:
        player_a_process.restart_process_and_children()
        success_a, message_a = player_a_process.run_timed_constructor(
            game_board, init_timeout, extra_ret_time
        )
        player_a_process.pause_process_and_children()

        player_b_process.restart_process_and_children()
        success_b, message_b = player_b_process.run_timed_constructor(
            game_board, init_timeout, extra_ret_time
        )
        player_b_process.pause_process_and_children()

    if not success_a and not success_b:
        game_board.set_winner(ResultArbiter.TIE, WinReason.FAILED_INIT)
        terminate_game(
            player_a_process, player_b_process, queues, out_queue, stop_event
        )
        return game_board, message_a, message_b, [], []
    elif not success_a:
        game_board.set_winner(ResultArbiter.PLAYER_B, WinReason.FAILED_INIT)
        terminate_game(
            player_a_process, player_b_process, queues, out_queue, stop_event
        )
        return game_board, message_a, message_b, [], []
    elif not success_b:
        game_board.set_winner(ResultArbiter.PLAYER_A, WinReason.FAILED_INIT)
        terminate_game(
            player_a_process, player_b_process, queues, out_queue, stop_event
        )
        return game_board, message_a, message_b, [], []

    # start actual gameplay
    #
    timer = 0
    winner = ResultArbiter.TIE
    while (
        # game_board.turn_count < 2 * game_board.MAX_TURNS
        not game_board.is_game_over()
    ):
        if display_game:
            init_display(game_board, "PLAYER A", "PLAYER B")

        if display_game:
            print_board(
                game_board,
                trapdoor_locations,
                clear_screen,
            )

        # The game board has already set the proper player as checking_player
        player_location = game_board.chicken_player.get_location()
        samples = trapdoor_manager.sample_trapdoors(player_location)

        if game_board.is_as_turn:
            # run a's turn
            player_label = "A"
            player_a_process.restart_process_and_children()
            moves, timer, message_a = player_a_process.run_timed_play(
                game_board, samples, game_board.player_time, extra_ret_time
            )
            player_a_process.pause_process_and_children()

        else:
            # run b's turn
            player_label = "B"
            player_b_process.restart_process_and_children()
            moves, timer, message_b = player_b_process.run_timed_play(
                game_board, samples, game_board.player_time, extra_ret_time
            )
            player_b_process.pause_process_and_children()

        if game_board.get_winner() is None:
            if moves is None:
                if timer == -1:
                    game_board.set_winner(Result.ENEMY, WinReason.CODE_CRASH)
                elif timer == -2:
                    game_board.set_winner(Result.ENEMY, WinReason.MEMORY_ERROR)
                else:
                    game_board.set_winner(Result.ENEMY, WinReason.TIMEOUT)
                game_board.is_as_turn = not game_board.is_as_turn
            else:
                dir, move_type = moves

                valid = game_board.apply_move(dir, move_type, timer=timer)
                
                if not valid:
                    message_b = "{moves}"
                    game_board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
                    game_board.is_as_turn = not game_board.is_as_turn
                elif game_board.player_time <= 0:
                    game_board.set_winner(Result.ENEMY, WinReason.TIMEOUT)

                # Check for trapdoor at new location
                new_location = game_board.chicken_player.get_location()

                if trapdoor_manager.is_trapdoor(new_location):
                    game_board.chicken_player.reset_location()
                    print(
                        f"Triggered trapdoor at {new_location}, {player_label} returned to {game_board.chicken_player.get_location()}"
                    )
                    game_board.chicken_enemy.increment_eggs_laid(
                        -1 * game_board.game_map.TRAPDOOR_PENALTY
                    )
                    game_board.found_trapdoors.add(new_location)
                    game_board.get_history().record_trapdoor(True, game_board.chicken_player.get_location())
                else:
                    game_board.get_history().record_trapdoor(False, game_board.chicken_player.get_location())
            # hack to deal with apply_move shenanigans


        if not moves is None and display_game:
            print_moves(not game_board.is_as_turn, moves, timer)
            time.sleep(delay)

        if not game_board.is_game_over():
            game_board.reverse_perspective()

    win_result = game_board.get_winner()

    # might seem reversed but this is correct due to the way apply_move works
    if game_board.is_as_turn:
        if win_result == Result.PLAYER:
            winner = ResultArbiter.PLAYER_B
        elif win_result == Result.ENEMY:
            winner = ResultArbiter.PLAYER_A
    else:
        if win_result == Result.PLAYER:
            winner = ResultArbiter.PLAYER_A
        elif win_result == Result.ENEMY:
            winner = ResultArbiter.PLAYER_B

    game_board.set_winner(winner, game_board.win_reason)

    if game_board.is_game_over():
        if display_game:
            print(f"{winner.name} wins by {game_board.get_win_reason().name}")

    terminate_game(player_a_process, player_b_process, queues, out_queue, stop_event)
    return game_board, trapdoor_locations, spawns, message_a, message_b


# closes down player processes
def terminate_game(process_a, process_b, queues, out_queue, stop_event):
    delete_module("player_a" + "." + "agent")
    delete_module("player_a")
    delete_module("player_b" + "." + "agent")
    delete_module("player_b")

    if not stop_event is None:
        stop_event.set()
        try:
            while True:
                print(out_queue.get_nowait())
        except:
            pass

    process_a.terminate_process_and_children()
    process_b.terminate_process_and_children()

    for q in queues:
        try:
            while True:
                q.get_nowait()
        except:
            pass
