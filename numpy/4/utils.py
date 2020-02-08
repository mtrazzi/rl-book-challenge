def print_transitions(env):
    for s_p in env.states:
        for r in env.r:
            for a in env.moves:
                for s in env.states:
                    if env.p(s_p, r, s, a) > 0:
                        print(f"p({s_p}, {r} | {s}, {a}) = {env.p(s_p, r, s, a)}")