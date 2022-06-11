"""Main survey file.
"""
import base64
import io
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from conditional_inference.bayes import Improper, Normal
from conditional_inference.stats import nonparametric
from flask_login import current_user
from hemlock import User, Page
from hemlock.functional import compile, validate, test_response
from hemlock.questions import Check, Input, Label, Range, Select, Textarea
from hemlock.utils.random import Assigner
from hemlock.utils.statics import make_figure
from scipy.stats import binom, f as f_distribution
from sqlalchemy_mutable.utils import partial

from src.results_text import megastudies
from src.clean_nudgeunits import SIMULATION_DATA_DIR

MIN_OBS, MAX_OBS = 500, 1500
MIN_TREATMENTS, MAX_TREATMENTS = 20, 40
NOT_ENOUGH_INFO = (
    "no_info",
    "The authors do not give enough information to answer this question",
)
REQUIRE_RESPONSE_FEEDBACK = "Please respond to this question."
N_RATINGS = 10
ORANGE = sns.color_palette().as_hex()[1]

sns.set()
assigner = Assigner({"bayes": (0, 1), "reading_comprehension_first": (0, 1)})
with open(os.path.join(SIMULATION_DATA_DIR, "control.json"), "r") as f:
    control_takeup_samples = json.load(f)

with open(os.path.join(SIMULATION_DATA_DIR, "prior.json"), "r") as f:
    effect_dist = nonparametric(json.load(f))


@User.route("/survey")
def seed():
    """Creates the main survey branch.

    Returns:
        List[Page]: List of pages shown to the user.
    """
    return [
        Page(Label("CONSENT FORM HERE")),
        Page(
            Label(
                """
                The next page will take 5-15 seconds to load after you click ">>".
                Please do not refresh the page.
                """
            ),
            navigate=make_simulation_branch,
        ),
    ]


def make_simulation_branch(origin):
    def plot_results(results, **kwargs):
        fig, ax = plt.subplots()
        results.point_plot(ax=ax, **kwargs)
        ax.axvline(control_takeup, linestyle="--")
        ax.set_ylabel(treatment_type.capitalize())
        ax.set_xlabel(f"{dep_variable.capitalize()} rate")
        return fig, ax

    def plot_to_html(fig, ax):
        estimates_figure = figure_to_html(fig)
        sns.scatterplot(
            x=takeup[::-1],
            y=np.arange(1, n_treatments + 1),
            color=ORANGE,
            marker="x",
            s=50,
            label=f"True {dep_variable} rates",
            ax=ax,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        return estimates_figure, figure_to_html(fig)

    def figure_to_html(fig):
        fig.savefig(buffer := io.BytesIO(), bbox_inches="tight")
        src = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        return make_figure(src, figure_align="center").replace("\n", "")

    # randomly generate the experiment parameters, e.g., number of observations per
    # treatment, number of treatments, treatment types, dependent variable, etc.
    assignment = assigner.assign_user()
    current_user.meta_data["n_obs"] = n_obs = int(np.random.uniform(MIN_OBS, MAX_OBS))
    current_user.meta_data["n_treatments"] = n_treatments = int(
        np.random.uniform(MIN_TREATMENTS, MAX_TREATMENTS)
    )
    current_user.meta_data["reverse_variability"] = reverse_variability = int(
        random.random() < 0.5
    )
    (
        current_user.meta_data["megastudy"],
        dep_variable,
        treatment_type,
        results_text,
        prediction_text,
    ) = random.choice(megastudies)
    prediction_text = prediction_text.replace("\n", "").strip()

    # simulate the experiment
    control_takeup = random.choice(control_takeup_samples)
    effects = effect_dist.rvs(size=n_treatments)
    takeup = np.clip(effects + control_takeup, 0.01, 0.99)
    simulate_takeup = lambda p: binom.rvs(n_obs, p) / n_obs
    takeup_estimates = np.apply_along_axis(simulate_takeup, 0, takeup)
    argsort = (-takeup_estimates).argsort()
    takeup = takeup[argsort]
    takeup_estimates = takeup_estimates[argsort]

    # test the null hypothesis
    df_explained = n_treatments - 1
    explained_var = n_obs * ((takeup_estimates - takeup_estimates.mean()) ** 2).sum()
    df_unexplained = n_treatments * (n_obs - 1)
    unexplained_var = (
        n_obs
        * (
            takeup_estimates * (1 - takeup_estimates) ** 2
            + (1 - takeup_estimates) * takeup_estimates**2
        ).sum()
    )
    total_obs_text = "{:,}".format(n_treatments * n_obs)
    n_obs_text = "{:,}".format(n_obs)
    f_stat = (explained_var / df_explained) / (unexplained_var / df_unexplained)
    pvalue = 1 - f_distribution.cdf(f_stat, df_explained, df_unexplained)

    # get OLS and Bayesian estimates
    cov = np.diag(takeup_estimates * (1 - takeup_estimates) / n_obs)
    ols_results = Improper(takeup_estimates, cov).fit(
        title="OLS estimates", n_samples=2
    )
    ols_fig, ols_ax = plot_results(ols_results)
    bayes_results = Normal(takeup_estimates, cov).fit(
        title="Bayesian estimates", n_samples=2
    )
    bayes_fig, bayes_ax = plot_results(bayes_results, robust=True, fast=True)
    xlim = min(ols_ax.get_xlim()[0], bayes_ax.get_xlim()[0]), max(
        ols_ax.get_xlim()[1], bayes_ax.get_xlim()[1]
    )
    ols_ax.set_xlim(xlim)
    bayes_ax.set_xlim(xlim)
    ols_estimates_plot, ols_true_plot = plot_to_html(ols_fig, ols_ax)
    bayes_estimates_plot, bayes_true_plot = plot_to_html(bayes_fig, bayes_ax)
    if assignment["bayes"]:
        estimates_plot, true_plot = bayes_estimates_plot, bayes_true_plot
        params = bayes_results.params
    else:
        estimates_plot, true_plot = ols_estimates_plot, ols_true_plot
        params = ols_results.params

    best_estimated_effect = "{:0.1f}".format(100 * (params[0] - control_takeup))
    best_effect = "{:.1f}".format(100 * (takeup[0] - control_takeup))
    best_treatment_was_truly_best = takeup[0] == takeup.max()

    results_text = results_text.format(
        n_treatments=n_treatments,
        total_obs_text=total_obs_text,
        regression_type="Bayesian"
        if assignment["bayes"]
        else "ordinary least squares (OLS)",
        estimates_plot=estimates_plot,
        rejected="rejected" if pvalue < 0.05 else "failed to reject",
        f_stat=f_stat,
        pvalue_text="P<.001" if pvalue < 0.001 else "P={:.3f}".format(pvalue),
        best_estimated_effect=100 * (params[0] - control_takeup),
    )

    treatment_variability_text = get_treatment_variability_text(
        treatment_type, reverse_variability
    )

    reading_comprehension_page = Page(
        Check(
            """
            We will now ask you some reading comprehension questions. Remember,
            these questions are about what the authors are communicating, **not**
            your opinions or interpretation.

            Please check the correct option below. On this page, we are asking
            you...
            """,
            shuffle(
                (1, "what the authors are communicating"),
                (0, "how you would interpret these results"),
            ),
            variable="reading_comprehension_check",
            validate=validate.compare_response(
                1, "Please select the correct response."
            ),
        ),
        Label(
            f"""
            The authors reported their results as follows:
            
            {results_text}
            """
        ),
        Check(
            "The authors tested the null hypothesis that...",
            shuffle(
                (0, f"all of the {treatment_type}s had different effects"),
                (1, f"all of the {treatment_type}s had the same effect"),
            )
            + [NOT_ENOUGH_INFO],
            variable="null_hypothesis_definition",
            validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
        ),
        Check(
            "Did the authors reject the null hypothesis?",
            shuffle("Yes", "No") + [NOT_ENOUGH_INFO],
            variable="null_hypothesis_rejected",
            validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
        ),
        Check(
            f"According to the authors, {prediction_text[0].lower() + prediction_text[1:]}",
            shuffle(
                ("more", f"more than {best_estimated_effect} percentage points"),
                ("less", f"less than {best_estimated_effect} percentage points"),
                ("equal", f"roughly {best_estimated_effect} percentage points"),
            )
            + [NOT_ENOUGH_INFO],
            variable="effect_size_implication",
            validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
        ),
    )
    interpretation_page = Page(
        Check(
            """
            We will now ask you some questions about how you would interpret the
            results.

            Please check the correct option below. On this page, we are asking
            you...
            """,
            shuffle(
                (0, "what the authors are communicating"),
                (1, "how you would interpret these results"),
            ),
            variable="interpretation_check",
            validate=validate.compare_response(
                1, "Please select the correct response."
            ),
        ),
        Label(
            f"""
            The authors reported their results as follows:
            
            {results_text}
            """
        ),
        effect_size_prediction := Input(
            prediction_text,
            variable="effect_size_prediction",
            validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
            append="percentage points",
            input_tag={"type": "number", "step": "any"},
        ),
        best_treatment_prediction := Input(
            f"""
            From 0-100%, how likely is it that the best-performing {treatment_type}
            is truly the most effective?
            """,
            variable="best_treatment_prediction",
            validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
            append="%",
            input_tag={"type": "number", "min": 0, "max": 100},
        ),
        treatment_variability := Select(
            f"""
            Rate your agreement with the following statement:
            
            **{treatment_variability_text}**
            """,
            get_agreement_ratings(reverse_variability),
            variable="treatment_variability_estimated",
            validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
        ),
    )
    return [
        Page(
            Label(
                """
                Thank you for participating in our study! We want to know how you, as
                a member of the behavioral science community, comprehend and interpret
                the results of behavioral science papers.

                To study this, we will show you the results of a hypothetical behavioral
                science study and ask you some questions about them. We are interested
                in what you understand the authors to be communicating about their
                results (reading comprehension) and how you interpret the results
                (interpretation/opinion). Therefore, we will ask you questions in two
                parts.

                **Reading comprehension.** What are the authors communicating?

                **Interpretation.** How would you interpret these results?
                """
            )
        ),
        *(
            (reading_comprehension_page, interpretation_page)
            if assignment["reading_comprehension_first"]
            else (interpretation_page, reading_comprehension_page)
        ),
        Page(
            Label(
                compile=partial(
                    get_true_label,
                    dep_variable,
                    treatment_type,
                    n_obs_text,
                    true_plot,
                    effect_size_prediction,
                    best_effect,
                    best_treatment_prediction,
                    best_treatment_was_truly_best,
                )
            ),
            Select(
                compile=partial(
                    get_treatment_variability_label,
                    dep_variable,
                    treatment_type,
                    treatment_variability,
                ),
                choices=get_agreement_ratings(reverse_variability),
                variable="treatment_variability_true",
                validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
            ),
        ),
        Page(
            Label(
                f"""
                Below, we plot the true {dep_variable} rates for each {treatment_type}
                (the <span style="color:{ORANGE}">orange x's</span>) alongside ordinary
                least squares estimates (top plot) and Bayesian estimates (bottom plot).
                {ols_true_plot}
                {bayes_true_plot}
                """
            ),
            Check(
                f"""
                Which results would give you a more accurate understanding of the true
                {dep_variable} rates if you saw them in a publication?
                """,
                shuffle((0, "Ordinary least squares results"), (1, "Bayesian results")),
                variable="bayes_more_accurate",
                validate=validate.require_response(REQUIRE_RESPONSE_FEEDBACK),
            ),
        ),
        Page(Label("We have recorded your responses. Thank you for participating!")),
    ]


def shuffle(*args):
    arr = list(args)
    random.shuffle(arr)
    return arr


def get_agreement_ratings(reverse=False):
    if reverse:
        convert_rating = lambda x: N_RATINGS - x
    else:
        convert_rating = lambda x: x
    ratings = [[convert_rating(i), str(i)] for i in range(N_RATINGS + 1)]
    ratings[0][1] += " Completely disagree"
    ratings[-1][1] += " Completely agree"
    ratings[int(N_RATINGS / 2)][1] += " Neither agree nor disagree"
    return [(None, "")] + [tuple(rating) for rating in ratings]


def get_true_label(
    question,
    dep_variable,
    treatment_type,
    n_obs_text,
    true_plot,
    effect_size_prediction,
    best_effect,
    best_treatment_prediction,
    best_treatment_was_truly_best,
):
    # fmt: off
    question.label = (
        f"""
        The results we just showed you were based on a simulated experiment. We
        started with the "true" {dep_variable} rates, marked by
        <span style="color:{ORANGE}"> orange x's</span> in the plot below. We then
        simulated randomly assigning {n_obs_text} people to each treatment condition.
        {true_plot}

        You estimated that the best-performing {treatment_type} would increase
        {dep_variable} rates by **{effect_size_prediction.response} percentage points**.
        In fact, the best-performing {treatment_type} increased {dep_variable} rates by
        **{best_effect} percentage points**.

        You also estimated that there was a **{best_treatment_prediction.response}%
        chance that the best-performing {treatment_type} was truly the most effective**.
        In fact, the best-performing {treatment_type}
        **was{" " if best_treatment_was_truly_best else " not"}** the most effective.
        """
    )
    # fmt: on


def get_treatment_variability_text(treatment_type, reverse):
    if reverse:
        return f"The {treatment_type}s tested in this study are more or less equally effective."

    # fmt: off
    return (
        f"""
        The {treatment_type}s tested in this study have very different effects. That is,
        some {treatment_type}s are much more effective than others.
        """
    ).replace("\n", "").strip()
    # fmt: on


def get_treatment_variability_label(
    question, dep_variable, treatment_type, treatment_variability
):
    reverse = current_user.get_meta_data()["reverse_variability"]
    treatment_variability_text = get_treatment_variability_text(treatment_type, reverse)
    response = (
        N_RATINGS - treatment_variability.response
        if reverse
        else treatment_variability.response
    )
    # fmt: off
    question.label = (
        f"""
        On the previous page, you rated your agreement with the statement,
        "{treatment_variability_text}" as a **{response} out of {N_RATINGS}**. Looking
        at the true {dep_variable} rates (the
        <span style="color:{ORANGE}">orange x's</span> in the plot above), rate your
        agreement with this statement again:

        **{treatment_variability_text}**
        """
    )
    # fmt: on
