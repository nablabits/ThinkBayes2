import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde


class BayesPredictions:
    def __init__(self, initial_history=4, risk=1):
        """
        Class constructor.

        Args:
            * initial_history: an int that tells us the point in historical records where we start
            predicting.
            * risk: a factor that lets increase slightly the optimal bid to increase the chances of
            winning both showcases.

        """
        self.initial_history = initial_history
        self.df = self.get_main_df()

        self.risk = risk

        # Compute price and off ranges
        self.price_range = np.linspace(10_000, 80_000, 1401)  # min showcase 18.3k, max 71.6k
        self.off_ranges = self.get_off_ranges()

        self.outcomes = pd.DataFrame({
            "p1_wins": [0.],
            "p2_wins": [0.],
            "no_one_wins": [0.]
        })

    def fix_data(self, player):
        """Determine the values for the current estimation."""
        assert player in (1, 2)

        bid = "bid_1"
        estimated_bid = "estimated_bid1"
        showcase = "showcase_1"
        estimated_diff = "estimated_diff1"
        opponent_error = "estimated_diff2"
        if player == 2:
            bid = "bid_2"
            estimated_bid = "estimated_bid2"
            showcase = "showcase_2"
            estimated_diff = "estimated_diff2"
            opponent_error = "estimated_diff1"

        estimated_row = self.df[self.df[estimated_bid].isna()].index[0]
        last_full_row = estimated_row - 1
        self.current_bid = self.df.loc[estimated_row, bid]
        self.showcase_history = self.df.loc[:last_full_row, showcase]
        self.error_history = self.df.loc[:last_full_row, estimated_diff]
        self.opponent_error_history = self.df.loc[:last_full_row, opponent_error]

    @staticmethod
    def read_historical_data(filename):
        """Read the showcase price data from the csv."""
        df = pd.read_csv(filename, index_col=0, skiprows=[1])
        df = df.dropna().transpose()
        df = df.rename(columns={
            'Showcase 1': 'showcase_1',
            'Showcase 2': 'showcase_2',
            'Bid 1': 'bid_1',
            'Bid 2': 'bid_2',
            'Difference 1': 'diff_1',
            'Difference 2': 'diff_2',
        })
        # it turns out that diffs are swapped
        df.loc[:, 'diff_1'] = -df.diff_1
        df.loc[:, 'diff_2'] = -df.diff_2
        return df

    def create_base_df(self):
        """Create a dataframe out of the outcomes of historical data."""
        df2011 = self.read_historical_data('data/showcases.2011.csv')
        df2012 = self.read_historical_data('data/showcases.2012.csv')
        return pd.concat([df2011, df2012], ignore_index=True)

    def get_main_df(self):
        """
        Compute the main dataframe.

        The main dataframe contains both historical data and the predictions to compare them.
        """
        base_df = self.create_base_df()
        base_df["estimated_bid1"] = base_df.bid_1
        base_df["estimated_diff1"] = base_df.diff_1
        base_df["estimated_bid2"] = base_df.bid_1
        base_df["estimated_diff2"] = base_df.diff_2
        base_df.iloc[self.initial_history:, 6:] = np.nan
        return base_df

    def get_off_ranges(self):
        """
        Compute an array of how off you fall from the real price for each price in the price_range.

        Returns: a 1401x1401 array with the offset from the real price where each real price is a
        price in the price_range.

        """
        _, vy = np.meshgrid(self.price_range, self.price_range)
        return self.price_range - vy

    def bayes_estimator(self):
        """
        Compute the posterior probability.

        Returns: a 1041x1 array with the posterior probability.

        """
        prior = gaussian_kde(self.showcase_history).pdf(self.price_range)
        likelihood = norm(self.current_bid, self.error_history.std()).pdf(self.price_range)
        posterior = prior * likelihood
        return posterior / posterior.sum()

    def get_chances_of_winning(self, off_range):
        """
        Compute the probability of winning depending on your opponent performance.

        Args:
            * off_range: a price_range array with 1401 values representing how off you get from the
             real price.

        Returns: a 1401x1 array with the probability of winning for each price in the price_range.
        """
        chances = off_range.copy()

        # if you overbid, you loose
        chances[off_range > 0] = 0

        # if your opponent overbids, you win
        opponent_overbids = (self.opponent_error_history > 0).mean()

        # Alternatively, if you underbid by less than your opponent, you win
        vx, vy = np.meshgrid(self.opponent_error_history, off_range)
        opponent_underbids_by_more = (vx < vy).mean(axis=1)

        chances_under_0 = opponent_overbids + opponent_underbids_by_more

        # increase the optimal bid to get more chances of winning both showcases
        chances_under_0[(off_range <= 250) & (off_range <= 0)] *= self.risk

        chances[off_range <= 0] = chances_under_0[off_range <= 0]
        return chances

    def estimate_winning_bid(self, player):
        """
        Get the bid that maximizes the chances of winning.

        The winning bid is the weighted sum of each chance range, that in turn represents the
        chances of winning for each price in the price_range. That weight is determined by the
        bayes estimator.

        Returns: an int that represents the optimal bid.
        """
        self.fix_data(player)
        bt = self.bayes_estimator()
        chances_ranges = np.apply_along_axis(self.get_chances_of_winning, 1, self.off_ranges)

        winning_bid_distribution = (chances_ranges.T * bt).sum(axis=1)
        winning_bid_distribution /= winning_bid_distribution.sum()
        s0 = pd.Series(winning_bid_distribution, index=self.price_range)
        return s0.idxmax()

    def compute_estimation(self):
        """Save the estimation onto the dataframe."""
        estimate = self.estimate_winning_bid(player=1)
        estimated_row = self.df[self.df.estimated_bid1.isna()].index[0]
        real_price = self.df.loc[estimated_row, 'showcase_1']

        self.df.loc[estimated_row, 'estimated_bid1'] = estimate
        self.df.loc[estimated_row, 'estimated_diff1'] = estimate - real_price

        # and for player 2
        estimate = self.estimate_winning_bid(player=2)
        estimated_row = self.df[self.df.estimated_bid2.isna()].index[0]
        real_price = self.df.loc[estimated_row, 'showcase_2']

        self.df.loc[estimated_row, 'estimated_bid2'] = estimate
        self.df.loc[estimated_row, 'estimated_diff2'] = estimate - real_price

    def compute_outcome(self):
        p1_gt_p2 = self.df.estimated_diff1 > self.df.estimated_diff2
        p1_overbids = self.df.estimated_diff1 > 0  # 24%
        p2_overbids = self.df.estimated_diff2 > 0  # 28%
        both_underbid = (~p1_overbids & ~p2_overbids)  # 53%
        both_overbid = (p1_overbids & p2_overbids)  # 6.7%

        p1_underbids_by_less = (p1_gt_p2 & both_underbid)  # 25.6%
        p2_underbids_by_less = (~p1_gt_p2 & both_underbid)  # 27.8%

        # Players win when they don't overbid AND either their opponent overbids OR they underbid
        # by less when their opponent does not overbid
        p1_wins = ((p1_underbids_by_less | p2_overbids) & ~p1_overbids).mean()  # 54% * 76% = 41%
        p2_wins = ((p2_underbids_by_less | p1_overbids) & ~p2_overbids).mean()  # 52% * 72% = 37%
        no_winner = both_overbid.mean()

        results = (p1_wins, p2_wins, no_winner)
        self.save_outcome(results)
        self.print_outcome(results)

    def save_outcome(self, results):

        # Compute the winner
        try:
            last_computed_idx = self.df[self.df.estimated_bid1.isna()].index[0] - 1
        except IndexError:
            last_computed_idx = self.df.index[-1]
        p1, p2 = self.df.loc[last_computed_idx, ["estimated_diff1", "estimated_diff2"]]

        index = self.outcomes.index.max() + 1

        # Both overbid
        if p1 > 0 and p2 > 0:
            self.outcomes.loc[index, "winner"] = 0
            print("No one wins")

        # p1 underbids by less or p2 overbids while p1 is underbidding
        elif (p1 > p2 or p2 > 0) and p1 <= 0:
            self.outcomes.loc[index, "winner"] = 1
            print("player 1 wins!")

        # p2 underbids by less or p1 overbids while p2 is underbidding
        elif (p2 > p1 or p1 > 0) and p2 <= 0:
            self.outcomes.loc[index, "winner"] = 2
            print("player 2 wins!")

        else:
            self.outcomes.loc[index, "winner"] = -1  # Error
            print("⚠️ Error ⚠️")

        self.outcomes.iloc[index, :3] = results

    @staticmethod
    def print_outcome(results):
        if np.random.uniform() > 9 / 10:  # print some results once in a while.
            print(50 * '*')
            p1_wins, p2_wins, no_winner = results
            print(f"player 1 wins: {100 * p1_wins.round(3)}% of the time")
            print(f"player 2 wins: {100 * p2_wins.round(3)}% of the time")
            print(f"No one wins: {100 * no_winner.round(3)}% of the time")

    def save_data_to_file(self):
        self.df.to_csv(f"09-outcomes/showcases-{self.initial_history}.csv")
        self.outcomes.to_csv(f"09-outcomes/outcomes-{self.initial_history}.csv")

    def run(self):
        leftover = 313 - self.initial_history
        for n in range(leftover):
            print(f"\nround {n}/{leftover - 1}")
            self.compute_estimation()
            self.compute_outcome()
        self.save_data_to_file()


class MakeUpP1(BayesPredictions):
    """
    A BayesPredictions variant where we can make up the guesses for p1.

    The intended use of this class is to compare how effective is our approach even if you are a
    bad guesser.
    """

    def __init__(self, worsen_factor, *args, **kwargs):
        self.worsen_factor = worsen_factor
        super().__init__(*args, **kwargs)

    def create_base_df(self):
        """
        Make up a df for player 1

        Player 1 is a Normal distribution centered at current real price times some worsen_factor
        (worse if it's higher than one).
        """
        df = super().create_base_df()
        df.loc[:, 'bid_1'] = np.random.normal(df.showcase_1 * self.worsen_factor, df.diff_1.std())
        df.loc[:, 'diff_1'] = df.bid_1 - df.showcase_1
        return df

    def save_data_to_file(self):
        f = str(self.worsen_factor).replace(".", "_")
        self.df.to_csv(f"09-outcomes/showcases-{self.initial_history}-{f}-worse.csv")
        self.outcomes.to_csv(f"09-outcomes/outcomes-{self.initial_history}-{f}-worse.csv")


if __name__ == '__main__':
    initial_history = 4

    # Comment one of the following, so it won't take very long to compute:
    BayesPredictions(initial_history).run()
    MakeUpP1(initial_history=initial_history, worsen_factor=1.3).run()
