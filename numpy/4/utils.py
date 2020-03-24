def print_old_psums(env):
    """Print sums of p(.,.|s,a) before normalization."""
    for s in env.states:
        for a in env.moves:
            p_sum = sum([env._p(s_p, r, s, a) for s_p in env.states
                        for r in env.r])
            print(f"sum of p(., .| {s}, {a}) = {p_sum}")
