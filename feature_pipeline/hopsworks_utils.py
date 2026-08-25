import os

import hopsworks
from dotenv import load_dotenv


def get_feature_store():
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"],
    )
    return project.get_feature_store()
