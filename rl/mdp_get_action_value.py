def get_action_value(mdp, state_values, state, action, gamma):
    """ Вычисляет Q(s,a) согласно формуле выше """
    q_func = 0
    
    for next_state, proba in mdp.get_next_states(state, action).items():
        reward = mdp.get_reward(state, action, next_state)
        v_func = state_values[next_state]
        q_func += proba * (reward + gamma * v_func)
    
    return q_func
