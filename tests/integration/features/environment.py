from tests.integration.facades.dataset import DatasetFacade
from tests.integration.facades.training import TrainingFacade
from tests.integration.mlabican_context import MlabicanContext

BEHAVE_DEBUG_ON_ERROR = False

def setup_debug_on_error(userdata):
    global BEHAVE_DEBUG_ON_ERROR
    BEHAVE_DEBUG_ON_ERROR = userdata.getbool("BEHAVE_DEBUG_ON_ERROR")


def before_all(context):
    setup_debug_on_error(context.config.userdata)


def before_scenario(context: MlabicanContext, scenario):
    context.dataset = None
    context.dataset_y_true = None
    context.predictions = None
    context.metrics = {}

    context.dataset_facade = DatasetFacade(context)
    context.training_facade = TrainingFacade(context)


def after_step(context: MlabicanContext, step):
    if BEHAVE_DEBUG_ON_ERROR and step.status == "failed":
        import ipdb
        ipdb.post_mortem(step.exc_traceback)
