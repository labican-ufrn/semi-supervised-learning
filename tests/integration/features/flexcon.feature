Funcionalidade: FlexCon

  Cenário: Algoritmo FlexCon-C classificando base com sucesso (#42)
    Dado que o algoritmo "FlexCon-C" foi instanciado com o classificador "naive"
    E que a base de dados "iris" possuía 30 por cento dos dados não rotulados
    Quando o algoritmo treinar
    Então a performance da métrica deverá ser maior a:
      | metric     | value  |
      | accuracy   | 0.9600 |
      | f1-score   | 0.9600 |
      | auc        | 0.9900 |
    E deverá ter acontecido, pelo menos, 50 iterações
