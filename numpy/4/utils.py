def trans_id(s_p, r, s, a):
    return str(s_p) + str(r) + str(s) + str(a)


def print_transitions(env, print_zeros=False):
    p_sum = 0
    for s in env.states:
        for a in env.moves:
            for s_p in env.states:
                for r in env.r:
                    proba = env.p[trans_id(s_p, r, s, a)]
                    if proba > 0 or (proba == 0 and print_zeros):
                        print(f"p({s_p},{r}|{s},{a}) = {proba}")
                        p_sum += proba


def print_psums(env):
    for s in env.states:
        for a in env.moves:
            p_sum = sum([env.p[trans_id(s_p, r, s, a)] for s_p in env.states
                        for r in env.r])
            print(f"sum of p(., .| {s}, {a}) = {p_sum}")


def print_old_psums(env):
    for s in env.states:
        for a in env.moves:
            p_sum = sum([env._p(s_p, r, s, a) for s_p in env.states
                        for r in env.r])
            print(f"sum of p(., .| {s}, {a}) = {p_sum}")
            if s == (0, 0) and a == 0:
                for s_p in env.states:
                    pr = sum([env._p(s_p, r, s, a) for r in env.r])
                    print(f"sum(p({s_p},.|{s},{a})) = {pr}")


def print_one_psum(env, s, a):
    p_sum = sum([env._p(s_p, r, s, a) for s_p in env.states
                        for r in env.r])
    print(*[f"p({s_p}, {r}| {s}, {a}) = {env._p(s_p, r, s, a)}"
            for s_p in env.states for r in env.r], sep='\n')
    print(f"sum of p(., .| {s}, {a}) = {p_sum}")
