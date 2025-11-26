import numpy as np
from game.board import Board
from game.enums import Cell, MoveType, WinReason
from game.history import History


def get_board_string(board: Board, trapdoors=set()):
    """
    Returns a string representation of the current state of the board.
    """

    main_list = []
    chicken_a = board.chicken_player if board.is_as_turn else board.chicken_enemy
    chicken_b = board.chicken_enemy if board.is_as_turn else board.chicken_player

    if board.is_as_turn:
        a_loc = board.chicken_player.get_location()
        a_eggs = board.eggs_player
        a_turds = board.turds_player
        b_loc = board.chicken_enemy.get_location()
        b_eggs = board.eggs_enemy
        b_turds = board.turds_enemy
    else:
        a_loc = board.chicken_enemy.get_location()
        a_eggs = board.eggs_enemy
        a_turds = board.turds_enemy
        b_loc = board.chicken_player.get_location()
        b_eggs = board.eggs_player
        b_turds = board.turds_player

    dim = board.game_map.MAP_SIZE
    main_list.append("  ")
    for x in range(dim):
        main_list.append(f"{x} ")
    main_list.append("\n")

    for y in range(dim):
        main_list.append(f"{y} ")
        for x in range(dim):
            current_loc = (x, y)
            if a_loc == current_loc:
                main_list.append("@ ")
            elif b_loc == current_loc:
                main_list.append("% ")
            elif current_loc in a_eggs:
                main_list.append("a ")
            elif current_loc in a_turds:
                main_list.append("A ")
            elif current_loc in b_eggs:
                main_list.append("b ")
            elif current_loc in b_turds:
                main_list.append("B ")
            elif current_loc in trapdoors:
                main_list.append("T ")
            else:
                main_list.append("  ")

        main_list.append("\n")

    return_string = "".join(main_list)
    return (
        return_string,
        chicken_a.get_eggs_laid(),
        chicken_b.get_eggs_laid(),
        chicken_a.get_turds_left(),
        chicken_b.get_turds_left(),
    )


def get_history_dict(board: Board,trapdoors=[],spawns = [[], []], errlog_a="", errlog_b=""):
    board_hist = board.history
    history_dict = {
        "pos": board_hist.pos,
        "left_behind_enums": board_hist.left_behind_enums,
        "a_eggs_laid": board_hist.a_eggs_laid,
        "b_eggs_laid": board_hist.b_eggs_laid,
        "a_turds_left": board_hist.a_turds_left,
        "b_turds_left": board_hist.b_turds_left,
        "a_time_left": board_hist.a_time_left,
        "b_time_left": board_hist.b_time_left,
        "a_moves_left": board_hist.a_moves_left,
        "b_moves_left": board_hist.b_moves_left,
        "trapdoor_triggered": board_hist.trapdoor_triggered,
    }

    left_behind = []
    for val in history_dict["left_behind_enums"]:
        match val:
            case MoveType.PLAIN:
                left_behind.append("plain")
            case MoveType.EGG:
                left_behind.append("egg")
            case MoveType.TURD:
                left_behind.append("turd")
            case _:
                left_behind.append("plain")
    history_dict["left_behind"] = left_behind
    history_dict.pop("left_behind_enums", None)

    history_dict.pop("map", None)

    history_dict["errlog_a"] = errlog_a
    history_dict["errlog_b"] = errlog_b

    history_dict["start_time"] = board.time_to_play
    history_dict["start_moves"] = board.MAX_TURNS

    history_dict["turn_count"] = board.turn_count
    history_dict["result"] = board.winner
    history_dict["reason"] = WinReason(board.win_reason).name
    history_dict["trapdoors"] = trapdoors

    
    history_dict["spawn_a"] = spawns[0]
    history_dict["spawn_b"] = spawns[1]
    

    return history_dict


def get_history_json(board: Board, trapdoors= [], spawns = [], err_a="", err_b=""):
    """
    Encodes the entire history of the game in a format readable by the renderer.
    """
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    return json.dumps(get_history_dict(board, trapdoors, spawns, err_a, err_b), cls=NpEncoder)
