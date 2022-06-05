"""Main survey file.
"""
import os

from flask_login import current_user
from hemlock import User, Page
from hemlock.functional import compile, validate, test_response
from hemlock.questions import Check, Input, Label, Range, Select, Textarea
from hemlock import utils
from sqlalchemy_mutable.utils import partial


@User.route("/survey")
def seed():
    """Creates the main survey branch.

    Returns:
        List[Page]: List of pages shown to the user.
    """
    return [
        Page(
            Label("Hello, world!")
        ),
        Page(
            Label("Goodbye, world!"),
            back=True
        )
    ]
