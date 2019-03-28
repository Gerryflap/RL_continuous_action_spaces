"""
    Models different matchmaking/ranking systems
"""
import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class MatchmakingSystem(ABC):
    @abstractmethod
    def report_outcome(self, pid1: int, pid2: int, outcome: int):
        """
        Reports the outcome of a game to the matchmaking/ranking system
        :param pid1: player_id of the first player
        :param pid2: player_id of the second player
        :param outcome: 0 on draw, 1 if p1 won, 2 if p2 won
        """
        pass

    @abstractmethod
    def get_matches(self, active_pids, max_matches=None) -> list:
        """
        Gets a number of new matches as an array of pid tuples
        :param active_pids: A set of active player ids
        :param max_matches: The maximum amount of matches to be retrieved or None if this doesn't matter
        :return: a List of pid tuples containing the new matches
        """
        pass

    @abstractmethod
    def get_rating(self, pid: int) -> float:
        pass


class RandomMatchMakingSystem(MatchmakingSystem):
    ratings = defaultdict(lambda: 0)

    def get_matches(self, active_pids, max_matches=None) -> list:
        active_pids = list(active_pids)

        if max_matches is not None:
            n_matches = min(len(active_pids)//2, max_matches)
        else:
            n_matches = len(active_pids) // 2
        other_players = active_pids[-n_matches:]
        random.shuffle(other_players)
        return list(zip(active_pids[:n_matches], other_players))

    def report_outcome(self, pid1: int, pid2: int, outcome: int):
        if outcome == 1:
            self.ratings[pid1] += 1
            self.ratings[pid2] -= 1
        elif outcome == 2:
            self.ratings[pid2] += 1
            self.ratings[pid1] -= 1

    def get_rating(self, pid: int) -> float:
        return self.ratings[pid]


class WLMatchMakingSystem(MatchmakingSystem):
    """
        Win - Loss Matchmaking. Tries to find close matches in win - loss number
    """
    ratings = defaultdict(lambda: 0)

    def get_matches(self, active_pids, max_matches=None) -> list:
        if len(active_pids) == 0:
            return []
        active_pids = list([(pid, self.get_rating(pid)) for pid in active_pids])
        random.shuffle(active_pids)

        if max_matches is not None:
            n_matches = min(len(active_pids)//2, max_matches)
        else:
            n_matches = len(active_pids) // 2

        active_pids = active_pids[:2*n_matches]
        active_pids = sorted(active_pids, key=lambda e: e[1])
        matches = []
        for (pid1, score1), (pid2, score2) in zip(active_pids[0::2], active_pids[1::2]):
            matches.append((pid1, pid2))

        return matches

    def report_outcome(self, pid1: int, pid2: int, outcome: int):
        if outcome == 1:
            self.ratings[pid1] += 1
            self.ratings[pid2] -= 1
        elif outcome == 2:
            self.ratings[pid2] += 1
            self.ratings[pid1] -= 1

    def get_rating(self, pid: int) -> float:
        return self.ratings[pid]


class ScaledMatchMakingSystem(MatchmakingSystem):
    """
        Scales rating gain/loss with rating difference
    """
    ratings = defaultdict(lambda: 0)

    def get_matches(self, active_pids, max_matches=None) -> list:
        if len(active_pids) == 0:
            return []
        active_pids = list([(pid, self.get_rating(pid) + np.random.normal(0, 1.5)) for pid in active_pids])
        random.shuffle(active_pids)

        if max_matches is not None:
            n_matches = min(len(active_pids)//2, max_matches)
        else:
            n_matches = len(active_pids) // 2

        active_pids = active_pids[:2*n_matches]
        active_pids = sorted(active_pids, key=lambda e: e[1])
        for _ in range(n_matches):
            # Get a random index between 0 and the max index - 1
            i = random.randint(0, n_matches*2 - 2)
            active_pids[i], active_pids[i+1] = active_pids[i+1], active_pids[i]
        matches = []
        for (pid1, score1), (pid2, score2) in zip(active_pids[0::2], active_pids[1::2]):
            matches.append((pid1, pid2))

        return matches

    def report_outcome(self, pid1: int, pid2: int, outcome: int):
        diff = self.get_rating(pid1) - self.get_rating(pid2)
        diff = max(min(diff, 200), -200)
        if outcome == 1:
            incr = min(1.05 ** -diff, 100)
            self.ratings[pid1] += 0.5*incr
            self.ratings[pid2] -= 0.5*incr
        elif outcome == 2:
            incr = min(1.05 ** diff, 100)
            self.ratings[pid2] += 0.5*incr
            self.ratings[pid1] -= 0.5*incr
        else:
            self.ratings[pid1] -= 0.05 * diff/2
            self.ratings[pid2] += 0.05 * diff/2

    def get_rating(self, pid: int) -> float:
        return self.ratings[pid]