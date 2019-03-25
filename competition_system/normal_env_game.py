from competition_system.competition_server import Client


def get_runner(env_builder, log=False):

    def run_game(client1: Client, client2: Client, result_queue):
        print("Game thread started with ", client1.pid, client2.pid)
        env = env_builder()
        s1, s2 = env.reset()
        done = False
        client1.send_srd(s1)
        client2.send_srd(s2)
        score1, score2 = 0, 0

        while not done:
            a1, a2 = client1.act_queue.get(), client2.act_queue.get()
            (new_state_1, new_state_2), (r1, r2), done, _ = env.step(a1, a2)
            client1.send_srd(new_state_1, r1, done)
            client2.send_srd(new_state_2, r2, done)
            score1 += r1
            score2 += r2

        if score1 == score2:
            # Assume a draw
            winner = 0
        elif score1 > score2:
            winner = 1
        else:
            winner = 2
        result_queue.put((client1.pid, client2.pid, winner))
    return run_game
