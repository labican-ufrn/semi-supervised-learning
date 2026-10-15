# ADR 001: Escolha de Framework BDD para Testes de Integração de ML

## Status
**Proposto**

## Contexto

O MLabican é uma biblioteca de aprendizado de máquina semissupervisionado baseada em algoritmos de *Self-Training*, comitês de classificadores e filtros de qualidade.  Atualmente, nossa suíte de testes unitários é construída utilizando a biblioteca nativa do Python, o `unittest`, pois evita utilizar dependências externas e simplifica a criação de testes. Com o avanço do desenvolvimento dos nossos modelos de Machine Learning (ML), identificamos a necessidade de implementar testes de integração baseados em Comportamento (BDD - Behavior-Driven Development).

O objetivo é validar pipelines de dados, inferências de modelos e integrações com bases vetoriais ou APIs externas usando linguagem natural, que é acessível para stakeholders de negócio e cientistas de dados (Gherkin).

Precisamos expandir nossa estratégia para testes de integração usando BDD.  Objetivando em validar o comportamento macro dos algoritmos (ex: fluxo completo de auto-treinamento, reavaliação de pseudo-rótulos e convergência do comitê com dados não rotulados) usando especificações em formato Gherkin (Dado/Quando/Então).

## Alternativas Consideradas
Nesta ADR serão comparados as bibliotecas `pytest-bdd` e `behave`.

### Opção 1: pytest-bdd
Extensão que traz os recursos de Gherkin para dentro do ecossistema do `pytest`.

* **Vantagens:**
  * **Executor Único (Unified Runner):** O `pytest` consegue descobrir e rodar nativamente nossos testes antigos escritos em `unittest`. Isso nos permite rodar toda a suíte do projeto com um único comando.
  * **Mocking e Ecossistema Avançado:** Acesso nativo ao plugin `pytest-mock` e ferramentas de paralelização (`pytest-xdist`), essenciais para simular endpoints pesados de ML (como HuggingFace ou OpenAI) e acelerar inferências.
  * **Gerenciamento de Estado via Fixtures:** Permite carregar pesos de modelos ou inicializar bancos de dados temporários uma única vez por sessão de teste, economizando memória e tempo de processamento.
* **Desvantagens:**
  * **Mudança de Paradigma, Alta Complexidade e Carga Cognitiva:** O `pytest` foca em injeção de dependência (fixtures), enquanto no `unittest` utiliza orientação a objetos (`setUp`/`tearDown`), que é mais simples de abstrair. Misturar as duas abordagens no mesmo repositório pode exigir uma curva de aprendizado.
  * **Verbosidade na Vinculação:** Exige declarar explicitamente a relação entre os cenários `.feature` e as funções Python no código.
  * **Curva de Aprendizado Dupla:** O time precisaria aprender uma nova ferramenta base (`pytest`), aprender a configurar plugins e aprender BDD ao mesmo tempo.
  * **Configuração Pesada:** Adiciona complexidade de configuração em um ambiente local que hoje depende apenas do `pre-commit`.

### Opção 2: behave
Framework BDD nativo e independente mais popular do ecossistema Python, muito similar ao Cucumber clássico.

* **Vantagens:**
  * **Fácil Integração com Pre-commit:** Pode ser adicionado facilmente como um gancho (`hook`) no `pre-commit` local, exigindo apenas a instalação do pacote via `pip`.
  * **Isolamento Total:** Roda de forma 100% independente do `unittest`. Não interfere em nada na estrutura de testes unitários que já temos construída.
  * **Sintaxe Limpa e Automação Gherkin:** Vincula os arquivos `.feature` aos passos em Python automaticamente, sem necessidade de código intermediário para amarrar os cenários. É mais intuitivo para cientistas de dados puristas e/ou iniciantes na programação.
  * **Simplicidade Absoluta e Isolamento:** Não exige conceitos avançados de engenharia de software (como injeção de dependência). A passagem de dados entre os passos do teste usa um objeto simples chamado `context` (funciona como um dicionário/sacola de variáveis).
* **Desvantagens:**
  * **Fragmentação de Ferramentas:** Seremos obrigados a manter dois comandos e pipelines separados no CI/CD (um para `unittest` e outro para o `behave`).
  * **Risco de Vazamento de Memória (Monalithic Context):** O compartilhamento de dados entre passos é feito por um objeto global chamado `context`. Passar estruturas pesadas (como DataFrames do Pandas ou tensores do PyTorch) por esse objeto global pode gerar efeitos colaterais e retenção alta de memória.
  * **Risco de Escalar com Código Desorganizado:** O objeto `context` pode virar uma "variável global" bagunçada se não houver revisão de código. Como estamos testando fluxos de *Self-Training* (que manipulam matrizes e dados rotulados), o revisor do PR precisará garantir que o `context` não está sendo sobrecarregado entre os testes.

## Decisão
A escolha recomendada para o MLabican é seguir com o **`behave`**.

**Justificativa:** A capacidade do `pytest-bdd` de reaproveitar o ecossistema do `pytest` (como controle granular de cache de dados e uso de fixtures para infraestrutura pesada) é interessante a primeira vista, pois proporciona muitas vantagens. Entretanto, a sua complexidade e carga coginitiva que será adicionada não supera os benefícios da sintaxe isolada do `behave`. Já o `behave` ofereça uma separação mais limpa e simples para programadores que vão começar a entender o conceito de testes.

Em um ambiente acadêmico e de formação de novos programadores, **a simplicidade e a separação didática superam o poder técnico**. Forçar alunos que nunca viram testes a entenderem o sistema de *fixtures* do `pytest` e os padrões do `unittest` simultaneamente geraria frustração e resistência à adoção de testes.

O `behave` oferece uma sintaxe direta, passos reutilizáveis intuitivos e valida de forma isolada os critérios de aceitação dos algoritmos semissupervisionados (ex: verificar se o modelo rotulou novos dados com base na confiança estabelecida). O problema de rodar dois comandos será mitigado automatizando ambos dentro do arquivo de configuração do nosso `pre-commit`.

## Consequências
* **Nova Dependência:** O projeto passará a ter o `behave` listado em suas dependências de desenvolvimento.
* **Automação no Pre-commit:** Atualizaremos o script de `pre-commit` para executar obrigatoriamente tanto a suíte de `unittest` (pre-commit) quanto a suíte de `behave` (pre-push) antes de permitir qualquer commit/push da branch local, assim garantimos qualidade do ambiente.
