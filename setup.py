from setuptools import setup, find_packages

with open('README.md', 'r') as arq:
    readme = arq.read()

setup(
    name="mlabican",
    version="1.1.0",
    license='MIT License',
    author="Luiz Miguel",
    description="Biblioteca de teste do Labican-UFRN",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="luiz.santos.090@ufrn.edu.br",
    keywords='mlabican',
    packages=find_packages(),
    install_requires=['scikit-learn', 'numpy'],
    python_requires=">=3.10",
)
