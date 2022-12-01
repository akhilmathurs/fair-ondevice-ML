import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fair_embedded_ml",
    version="0.0.1",
    author="Wiebke Toussaint",
    author_email="w.toussaint@tudelft.nl",
    description="Fair Embedded Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akhilmathurs/fair_embedded_ml",
    license="GPL-3.0-or-later",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
    ],
    packages=setuptools.find_packages(where="fair_embedded_ml"),
    py_modules=[
        'fair_embedded_ml.io_ops', 'fair_embedded_ml.models','fair_embedded_ml.train_and_eval', 
        'fair_embedded_ml.common_utilities', 'fair_embedded_ml.metrics', 'fair_embedded_ml.results_analysis',
        'fair_embedded_ml.results_plot'],
    entry_points={
        'console_scripts': [
            'train_and_eval = fair_embedded_ml.train_and_eval:cli',
        ]
    },
    python_requires=">=3.6",
)