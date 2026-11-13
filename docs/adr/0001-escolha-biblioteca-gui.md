# ADR 0001: Escolha da biblioteca para a GUI

## Status

**Aprovado**

## Contexto

A Issue #33 propõe uma interface gráfica para configurar e executar experimentos do MLabican. A Issue #34 avaliou bibliotecas considerando o tipo de aplicação, a criação de formulários, a execução de tarefas longas, a instalação e as licenças.

A equipe demonstrou preferência por uma aplicação desktop. A análise completa das bibliotecas está disponível na [Discussion #40](https://github.com/labican-ufrn/semi-supervised-learning/discussions/40). e na [Issue #34](https://github.com/labican-ufrn/semi-supervised-learning/issues/34).

## Alternativas consideradas

- **Tkinter:** simples e geralmente disponível com Python, mas tem aparência mais tradicional.
- **CustomTkinter:** baseado no Tkinter, com componentes de aparência mais moderna.
- **PySide6:** oferece recursos abrangentes para aplicações desktop, com maior complexidade de implementação.
- **Streamlit e Gradio:** permitem criar interfaces web acessíveis pelo navegador.

## Decisão

Adotar **CustomTkinter** para desenvolver a GUI do MLabican.

## Justificativa

A equipe prefere uma aplicação desktop. O CustomTkinter mantém uma abordagem semelhante à do Tkinter e oferece componentes com aparência mais moderna. A equipe discutiu as alternativas e concordou com essa escolha. Protótipos simples foram criados para validar a biblioteca e comparar sua apresentação com a do Tkinter padrão.

## Consequências

- A GUI terá uma dependência adicional, que deverá ser declarada no grupo opcional `gui` do `pyproject.toml`.
- Tarefas longas deverão ser executadas de modo que a interface continue responsiva e possa atualizar o progresso e os logs.
- A distribuição e a compatibilidade da aplicação desktop deverão ser consideradas durante a implementação.

## Referências

- [Issue #34 — Estudo e Seleção de Biblioteca para Interface Gráfica em Python](https://github.com/labican-ufrn/semi-supervised-learning/issues/34)
- [Discussão #40 — Avaliação de bibliotecas Python para a GUI](https://github.com/labican-ufrn/semi-supervised-learning/discussions/40)