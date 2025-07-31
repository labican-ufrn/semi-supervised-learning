from setuptools import setup, find_packages

with open('README.md', 'r') as arq:
    readme = arq.read()

setup(
    name="mlabican",
    version="1.1.1",
    license='MIT License',
    author="Luiz Miguel Santos Silva",
    author_email="luiz.santos.090@ufrn.edu.br",
    description="Algoritmos de Self-Training com estratégias de reavaliação de pseudo-rótulos para aprendizado semissupervisionado",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/mlabican/",
    keywords='mlabican',
    packages=find_packages(),
    install_requires=['scikit-learn', 'numpy'],
    python_requires=">=3.10",
)
