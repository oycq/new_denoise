def _get_round(x, round_x:int=8):
    x = int(round(x))
    x = (x // round_x) * round_x
    return x
