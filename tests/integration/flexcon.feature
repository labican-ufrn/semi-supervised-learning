Funcionalidade: FlexCon

  Cenário: Algoritmo FlexCon-C classificando base com sucesso (#42)
    Dado que o algoritmo "FlexCon-C" está instanciado com o classificador "naive"
    E que a base de dados "iris" possui "30" porcento dos dados não rotulados
    Quando o algoritmo treinar
    Então a performance da métrica deverá ser igual a:
      | metric     | value  |
      # | iterations |  60    |
      | accuracy   | 0.9667 |
      | f1-score   | 0.9666 |
      | auc        | 0.9976 |
