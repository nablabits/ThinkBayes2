import numpy as np
import pandas as pd


class Tournament:
    """Evolve two players who learn chess during a tournament."""

    # The factor by which we change the scores.
    K = {"main": 32, "master": 16}

    # the minimum score a player will ever have. See self.set_min()
    min_threshold = 100

    # Learning rate parameters, see self.learning_experience()
    winner_learning_rate = (0, 0.5)  # mu, std
    looser_learning_rate = (2, 1)
    draw_learning_rate = (0, 1)

    number_of_matches = 1000

    def __init__(
        self,
        player_a_score: int = 100,
        player_b_score: int = 100,
        discriminant: int = 400,
    ) -> None:
        self.player_a_score = player_a_score
        self.player_b_score = player_b_score
        self.discriminant = discriminant
        self.record_tape = pd.DataFrame(
            {},
            columns=[
                "prob_a",
                "prob_b",
                "match_outcome",
                "pre_match_score_a",
                "pre_match_score_b",
                "post_match_score_a",
                "post_match_score_b",
                "post_training_score_a",
                "post_training_score_b",
            ],
        )

    def prob_a_wins(self) -> float:
        """
        Get the probability of player_score A winning.

        Note that the logistic function will return 1 - p were we calculating the probability of B
        winning, that is, swapping the terms of d. Therefore, in practice we don't need a function
        for B.
        """
        d = self.player_b_score - self.player_a_score
        exp_term = 10 ** (d / self.discriminant)
        return 1 / (1 + exp_term)

    @staticmethod
    def get_draw(p: float) -> bool:
        """
        Simulate a possible draw.

        We need to simulate a draw as a possible outcome for a match. A draw is more likely when
        p=.5, so if p < .5 we will use that p to get a bernoulli trial, otherwise we have to
        compute the distance between 1 and the probability which will flip the increasing
        probability.
        """
        if p > 0.5:
            p = 1 - p  # this way p=.7 will become p=.3
        return np.random.binomial(1, p) == 1

    def play_match(self, p: float) -> float:
        """
        Simulate the outcome of a single match.

        There are 3 possible outcomes:
                1: player A wins
                0.5: it was a draw
                0: player B wins
        """
        if self.get_draw(p):
            return 0.5
        return np.random.binomial(1, p)

    def get_k_score_for_player(self, player_score: int) -> int:
        """
        Get the factor by which scores vary.

        Although there are more sophisticated ways to compute this factor, we will stick to it's
        simplest version that depends on the current score of the player_score.
        https://en.wikipedia.org/wiki/Elo_rating_system#The_K-factor_used_by_the_USCF
        """
        if player_score > 2299:
            return self.K["master"]
        return self.K["main"]

    def update_scores(self, pa: float, match_outcome_a: float) -> None:
        """Update the score for each player after a match depending on the outcome."""
        match_outcome_b = 1 - match_outcome_a
        pb = 1 - pa
        ka, kb = [
            self.get_k_score_for_player(player)
            for player in (self.player_a_score, self.player_b_score)
        ]
        self.player_a_score += ka * (pa - match_outcome_a)
        self.player_b_score += kb * (pb - match_outcome_b)
        self.set_min()

    def set_min(self) -> None:
        """
        Avoid players from going below self.min_threshold.

        Whenever we update, we can end up with negative amounts so let's set a minimum score for
        players.
        """
        if self.player_a_score < self.min_threshold:
            self.player_a_score = self.min_threshold
        if self.player_b_score < self.min_threshold:
            self.player_b_score = self.min_threshold

    def learning_experience(self, match_outcome_a: float) -> None:
        """
        Update the scores of individual players after a match.

        If we leave the scores as they are one player_score is going to grow at the expense of the
        other, as chess is a zero-sum game, so we need to train them pretending that they gain
        expertise by means other than stealing points to the opponent.

        In real life, though, what happens is as long as new players are added to the global
        ranking, the bag of available points gets bigger and bigger, so in a sense, the game
        becomes a positive sum game as a whole.

        The intuition behind this training is that after a match both players learn
        something in the process, but the player who losses, learns a bit more than the
        player who wins.
        """
        if match_outcome_a == 1:  # A wins
            a_parameters = self.winner_learning_rate
            b_parameters = self.looser_learning_rate
        elif match_outcome_a == 0:  # B wins
            a_parameters = self.looser_learning_rate
            b_parameters = self.winner_learning_rate
        else:  # draw
            a_parameters = self.draw_learning_rate
            b_parameters = self.draw_learning_rate

        mu_a, std_a = a_parameters
        mu_b, std_b = b_parameters
        self.player_a_score += np.random.normal(mu_a, std_a)
        self.player_b_score += np.random.normal(mu_b, std_b)
        self.set_min()

    def run(self) -> pd.DataFrame:
        """Start the simulation of the tournament."""
        for n in range(self.number_of_matches or 1):
            # play match
            pa = self.prob_a_wins()
            a_wins = self.play_match(pa)  # 1 for win, 0 for loss and .5 for draw
            self.record_tape.loc[
                n,
                [
                    "prob_a",
                    "prob_b",
                    "match_outcome",
                    "pre_match_score_a",
                    "pre_match_score_b",
                ],
            ] = [pa, 1 - pa, a_wins, self.player_a_score, self.player_b_score]

            # Update scores
            self.update_scores(pa, a_wins)
            self.record_tape.loc[n, ["post_match_score_a", "post_match_score_b"]] = [
                self.player_a_score,
                self.player_b_score,
            ]

            # Compute learning experience
            self.learning_experience(a_wins)
            self.record_tape.loc[
                n,
                [
                    "post_training_score_a",
                    "post_training_score_b",
                ],
            ] = [self.player_a_score, self.player_b_score]

        # Compute the total points in the tournament
        self.record_tape["total_points"] = (
            self.record_tape.post_training_score_a
            + self.record_tape.post_training_score_b
        )
        return self.record_tape
