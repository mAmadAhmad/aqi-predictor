import os

import hopsworks
from dotenv import load_dotenv


def get_project():
    load_dotenv()
    return hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"],
    )


def get_feature_store():
    return get_project().get_feature_store()
