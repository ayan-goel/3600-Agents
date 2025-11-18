
import numpy as np
class History:
    """
    Internal utility for recording history used only by the game runner. 
    Do not call these functions, they will not be helpful to you.
    """
    def __init__(self):
        self.pos=[]
        self.left_behind_enums=[]
        self.a_eggs_laid=[]
        self.b_eggs_laid=[]
        self.a_turds_left=[]
        self.b_turds_left=[]
        self.a_time_left=[]
        self.b_time_left=[]
        self.a_moves_left=[]
        self.b_moves_left=[]
        self.trapdoor_triggered=[]

    def record_trapdoor(self, trigger_trap):
        self.trapdoor_triggered.append(int(trigger_trap))


    def record_round_update(
            self, loc, move_type, eggs_laid_a, eggs_laid_b, 
            turds_left_a, turds_left_b, time_left_a, time_left_b,
            moves_left_a, moves_left_b, is_as_turn):
        self.pos.append(loc)
        if(is_as_turn):
            self.left_behind_enums.append(int(move_type))
            self.a_eggs_laid.append(eggs_laid_a)
            self.b_eggs_laid.append(eggs_laid_b)
            self.a_turds_left.append(turds_left_a)
            self.b_turds_left.append(turds_left_b)
            self.a_time_left.append(time_left_a)
            self.b_time_left.append(time_left_b)
            self.a_moves_left.append(moves_left_a)
            self.b_moves_left.append(moves_left_b)
        else:
            self.left_behind_enums.append(int(move_type))
            self.a_eggs_laid.append(eggs_laid_b)
            self.b_eggs_laid.append(eggs_laid_a)
            self.a_turds_left.append(turds_left_b)
            self.b_turds_left.append(turds_left_a)
            self.a_time_left.append(time_left_b)
            self.b_time_left.append(time_left_a)
            self.a_moves_left.append(moves_left_b)
            self.b_moves_left.append(moves_left_a)