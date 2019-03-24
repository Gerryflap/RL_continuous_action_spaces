"""
    Models different matchmaking/ranking systems
"""
from abc import ABC, abstractmethod
from collections import defaultdict


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
        n_matches = min(len(active_pids)//2, max_matches)
        return list(zip(active_pids[:n_matches], active_pids[-n_matches:]))

    def report_outcome(self, pid1: int, pid2: int, outcome: int):
        if outcome == 1:
            self.ratings[pid1] += 1
            self.ratings[pid2] -= 1
        elif outcome == 2:
            self.ratings[pid2] += 1
            self.ratings[pid1] -= 1

    def get_rating(self, pid: int) -> float:
        return self.ratings[pid]
