from behave import given, then, when

from tests.integration.mlabican_context import MlabicanContext


@given('que o algoritmo "{algorithm}" foi instanciado com o classificador "{model}"')
def create_algorithm(
    context: MlabicanContext,
    algorithm: str,
    model: str,
) -> None:
    context.training_facade.initialize(algorithm, model)


@given('que a base de dados "{dataset}" possuía {ssl_rate:n} por cento dos dados não rotulados')
def create_ssl_dataset(
    context: MlabicanContext,
    dataset: str,
    ssl_rate: int,
) -> None:
    context.dataset_facade.initialize(dataset, ssl_rate)


@when('o algoritmo treinar')
def model_fit(context: MlabicanContext) -> None:
    context.training_facade.train()


@then('a performance da métrica deverá ser maior a:')
def evaluate(context: MlabicanContext) -> None:
    metrics = [row['metric'] for row in context.table.rows]
    scores = context.training_facade.evaluate(metrics)

    for score, row in zip(scores, context.table.rows):
        assert score >= float(row['value']), (
            f'A métrica {row["metric"]} foi {score}, deveria ser {row["value"]}'
        )


@then('deverá ter acontecido, pelo menos, {min_iterations:d} iterações')
def check_iterations(context: MlabicanContext, min_iterations: int) -> None:
    assert context.training_facade.model.n_iter_ >= min_iterations, (
        f'Número de iterações foi {context.training_facade.model.n_iter_}, deveria ser pelo menos {min_iterations}'
    )
