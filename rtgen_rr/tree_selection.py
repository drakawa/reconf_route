import math
import numpy as np
import itertools as it

def dst_transpose(src: int, num_nodes: int) -> int:
    """destination for transpose traffic.

    Example: 
        0b0010 -> 0b1000 (2 -> 8) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^(2*i))

    Returns:
        (int): destination node id

    >>> dst_transpose(2, 16)
    8
    >>> dst_transpose(0b101010, 64)
    21
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes or lg % 2 != 0:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[(lg//2):] + src_bit[:(lg//2)], 2)

def dst_reverse(src: int, num_nodes: int) -> int:
    """destination for reverse traffic.

    Example: 
        0b0010 -> 0b0100 (2 -> 4) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (int): destination node id

    >>> dst_reverse(2, 16)
    4
    >>> dst_reverse(0b001110, 64)
    28
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[::-1], 2)

def dst_shuffle(src: int, num_nodes: int) -> int:
    """destination for shuffle traffic.

    Example: 
        0b1110 -> 0b1101 (14 -> 13) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (int): destination node id

    >>> dst_shuffle(14, 16)
    13
    >>> dst_shuffle(0b101110, 64)
    29
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[1:] + src_bit[:1], 2)

def gen_TM_from_tf(tf, num_nodes):
    """generate Traffic Matrix from traffic function

    Args:
        tf (Callable): traffic function (src: int, num_nodes: int -> int)
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (ndarray): num_nodes x num_nodes traffic matrix 
    """
    
    tm = np.zeros((num_nodes, num_nodes))
    for src in range(num_nodes):
        dst = tf(src, num_nodes)
        if src != dst:
            tm[src, dst] = 1

    return tm


if __name__ == "__main__":

    import doctest
    doctest.testmod()

    print(gen_TM_from_tf(dst_shuffle, 64))
    print(np.nonzero(gen_TM_from_tf(dst_shuffle, 64)))
