import os

import pytest
from hemlock import User, create_test_app
from hemlock.app import db

import src

DATA_DIR = "data"
N_USERS = 100


@pytest.fixture
def app():
    yield create_test_app()
    for user in User.query.all():
        db.session.delete(user)
    db.session.commit()


def test(app):
    User.test_multiple_users(N_USERS)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    User.get_all_data().to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
