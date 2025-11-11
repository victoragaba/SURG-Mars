import numpy as np

halfpi = np.pi/2
twopi = np.pi*2


def bound(params: list) -> list:
    
    turns = [halfpi, np.pi, twopi]
    
    # leave input unchanged
    out = params.copy()
    
    # driven by dip: case for every quadrant
    overturned = out[1] % turns[2] > turns[1]
    out[1] = out[1] % turns[1]
    
    # strike-dip relationship
    if out[1] > turns[0]:
        out[1] = turns[1] - out[1]
        out[0] += turns[1]
    out[0] = out[0] % turns[2]
    
    # bound the rake
    if overturned: out[2] += turns[1]
    out[2] = out[2] % turns[2]
    if out[2] > turns[1]:
        out[2] -= turns[2]
        
    return out