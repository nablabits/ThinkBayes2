import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal, norm


class PenguinClassifier:
    """Tell apart penguins by their features.

    Initially we will ask for the culmen_length to tell apart Adelie penguins
    and flipper_length for gentoo ones.
    """

    features = [
        "culmen_length_mm",
        "flipper_length_mm",
        "culmen_depth_mm",
        "body_mass_g",
    ]

    def __init__(
        self,
        culmen_length: int = None,
        flipper_length: int = None,
        culmen_depth: int = None,
        body_mass: int = None,
        naive: bool = True,
        data_source: str = "data/penguins_raw.csv",  # for the notebook
    ) -> None:
        self.df = self._get_df(data_source)
        self.culmen_length = culmen_length
        self.flipper_length = flipper_length
        self.culmen_depth = culmen_depth
        self.body_mass = body_mass
        self.naive = naive
        self.bayes_table = self._create_bayes_table()
        self.grid_of_probabilities = None

    @staticmethod
    def _get_df(data_source) -> pd.DataFrame:
        """Build an easy to work with dataframe off the datasource."""
        df = pd.read_csv(data_source)
        df.loc[:, "Species"] = df.Species.apply(lambda x: x.split()[0])
        df.columns = (
            df.columns.str.lower()
            .str.replace(" (o/oo)", "", regex=False)
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
        return df

    def _create_bayes_table(self) -> pd.DataFrame:
        """
        Create the bayes table that will make the predictions.

        Initially we just want to set the prior. Although we can set a uniform one 1/3, given that:
            penguins in the data are not evenly distributed, and
            we will do few updates
        we might want to assign different priors that reflect the fact that Chinstrap with 68 rows
        carries less information than Adelie with 152.
        """
        bayes_table = self.df.groupby("species").region.count().to_frame()
        bayes_table.rename(columns={"region": "prior"}, inplace=True)
        bayes_table /= bayes_table.sum()
        return bayes_table

    def show_pair_plot(self) -> None:
        """Display a pair plot of all the features."""
        cols = [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "species",
        ]
        sns.pairplot(self.df[cols], hue="species")

    def _run_naive_classifier(self) -> None:
        """
        Run a naive Classifier over the whole set of features.

        The intuition behind the naive classifier is that one can assume that the features are
        independent one another, which is not true. With this in mind we can run a bayesian update
        of the prior for each of the features. We will use a normal distribution with parameters
        matching the ones in the data source for the given specie as that seems to be their
        distribution.
        """

        def compute_likelihood(feature: str, data: int) -> None:
            """Compute the likelihood distribution for the given feature for all the species."""
            likes = dict()
            for species in self.df.species.unique():
                values = self.df[self.df.species == species][feature]
                likes[species] = norm(values.mean(), values.std()).pdf(data)
            like_series = pd.Series(likes, name="likes")
            self.bayes_table = self.bayes_table.join(like_series)

        def compute_posterior() -> None:
            """Compute the posterior distribution for all the species."""
            posterior = self.bayes_table.prior * self.bayes_table.likes
            posterior /= posterior.sum()
            self.bayes_table["posterior"] = posterior

        attrs = (
            self.culmen_length,
            self.flipper_length,
            self.culmen_depth,
            self.body_mass,
        )
        self.bayes_table = self._create_bayes_table()  # always start fresh

        initial = True
        for feat, att in zip(self.features, attrs):
            if not initial:
                # Once we have a valid posterior we want to start updating the prior.
                self.bayes_table.loc[:, "prior"] = self.bayes_table.posterior
                self.clear_bayes_table()
            compute_likelihood(feat, att)
            compute_posterior()
            initial = False

    def run_correlated_classifier(self) -> None:
        """
        Run a non naive classifier over the whole set of features.

        The intuition behind the non-naive classifier is that it accounts for correlated features.
        It feels more natural as looking at them one can notice that, for instance, an increase in
        culmen_length is related to an increase in flipper_length which makes sense as the body
        parts grow proportionally.

        Pretty much like we did for the naive classifier, we assume that the features are normally
        distributed, however in this case for the likelihood we will create a multivariate normal
        distribution that will account for the correlations between features.
        """
        likes = dict()
        for specie in self.df.species.unique():
            mu = self.df[self.df.species == specie][self.features].mean()
            cov = self.df[self.df.species == specie][self.features].cov()

            likes[specie] = multivariate_normal(mu, cov).pdf(
                (
                    self.culmen_length,
                    self.flipper_length,
                    self.culmen_depth,
                    self.body_mass,
                )
            )

        # update prior
        like_series = pd.Series(likes, name="likes")
        self.bayes_table = self.bayes_table.join(like_series)
        self.bayes_table["posterior"] = self.bayes_table.prior * self.bayes_table.likes
        self.bayes_table["posterior"] /= self.bayes_table.posterior.sum()

    def run_classifier(self) -> None:
        """Choose and run the right classifier depending on the naive attribute."""
        if self.naive:
            self._run_naive_classifier()
        else:
            self.run_correlated_classifier()

    def clear_bayes_table(self) -> None:
        """
        Restart the bayes table to be able to run new computations

        After each iteration computing accuracy or each step in the naive classifier, we need to
        drop last computed columns so `pd.join` won't hit an overlapping error.
        """
        df_cols = self.bayes_table.columns
        cols_to_drop = df_cols[df_cols != "prior"]
        self.bayes_table = self.bayes_table.drop(columns=cols_to_drop)

    def accuracy(self) -> float:
        """Compute the accuracy of the classifier over the whole dataset."""
        expected, predicted = list(), list()
        features = [
            "culmen_length_mm",
            "flipper_length_mm",
            "culmen_depth_mm",
            "body_mass_g",
        ]
        not_na = self.df[self.df[features].isna().sum(axis=1) == 0]
        for i, row in not_na.iterrows():
            (
                self.culmen_length,
                self.flipper_length,
                self.culmen_depth,
                self.body_mass,
            ) = row[features]
            self.run_classifier()
            expected.append(self.df.at[i, "species"])
            predicted.append(self.prediction)
            self.clear_bayes_table()
        return (np.array(expected) == np.array(predicted)).sum() / len(expected)

    @property
    def prediction(self) -> str:
        """Show the predicted value in the computations."""
        return self.bayes_table.posterior.idxmax()

    def run(self, print_=False) -> pd.DataFrame:
        """Compute a single prediction based on the class constructors."""
        self.run_classifier()
        if print_:
            print(
                f"{self.prediction} is the most likely species",
                f"{(self.bayes_table.posterior.max() * 100).round(2)}%",
            )

        return self.bayes_table


if __name__ == "__main__":
    accuracy = PenguinClassifier(
        naive=True, data_source="../data/penguins_raw.csv"
    ).accuracy()
    print(accuracy)
