from setuptools import setup, find_packages

setup(
    name="DECODE-DiseaseTrajectories",
    version="1.0.0",
    description="This package analyses temporal relationships between diseases by identifying"
                " statistically significant disease pairs and determining their temporal directionality"
                " through statistical testing. It then constructs disease trajectories by connecting"
                " these significant pairs in sequence, and employs a shortest-path graph-based clustering"
                " method to group similar disease patterns.",
    author="Rania Kousovista",
    author_email=" ",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
