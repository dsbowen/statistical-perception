megastudies = [
    (
        "voting",
        "voting",
        "letter",
        """
        *In this study, we sent one of {n_treatments} letters to {total_obs_text} registered voters in the United States encouraging them to vote in upcoming state and local elections. The letters used different psychological mechanisms to promote voting. For example, one letter reminded recipients that whether they cast a ballot is a matter of public record, while another told recipients that voting is their patriotic duty.*

        *We estimated the voting rates associated with each letter using {regression_type} regression. The figure below shows the results. The y-axis shows the different letters, and the x-axis shows the estimated voting rates. The blue vertical line is the voting rate in a control condition in which we did not send people a letter.*
        {estimates_plot}

        *We {rejected} the null hypothesis that all letters had the same effect using an ANOVA (F={f_stat:.2f}, P={pvalue:.3f}).*

        *We defined the treatment effect of each letter as the increase in voting rates compared to the control condition. Our best-performing letter increased voting rates by an estimated {best_estimated_effect:.01f} percentage points.*
        """,
        """
        If we sent the best-performing letter to many voters (from the same population the study sampled from), it would increase voting rates by...
        """,
    ),
    (
        "job_training",
        "completion",
        "nudge",
        """
        *In this study, we used one of {n_treatments} nudges to encourage {total_obs_text} unemployed people in the United States to complete a job training program, which required attending a four-week training course. The nudges used different psychological mechanisms to encourage attendance. For example, one nudge paid people a small amount of money each time they attended, while another helped people set reminders.*

        *We estimated the completion rates associated with each nudge using {regression_type} regression. The figure below shows the results. The y-axis shows the different nudges, and the x-axis shows the estimated completion rates. The blue vertical line is the completion rate in a control condition in which we did not use a nudge.*
        {estimates_plot}

        *We {rejected} the null hypothesis that all nudges had the same effect using an ANOVA (F={f_stat:.2f}, P={pvalue:.3f}).*

        *We defined the treatment effect of each nudge as the increase in completion rates compared to the control condition. Our best-performing nudge increased completion rates by an estimated {best_estimated_effect:.01f} percentage points.*
        """,
        """
        If we used the best-performing nudge to encourage many unemployed people (from the sample population the study sampled from) to complete a job training program, it would increase completion rates by...
        """,
    ),
]
