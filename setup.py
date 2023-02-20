# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


with open("fm/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()


setup(
    name="rna-fm",
    version=version,
    description="RNA Foundation Model (rna-fm): Pretrained language models for RNAs. From CUHK AIH Lab.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="CUHK AIH Lab",
    url="https://github.com/ml4bio/RNA-FM",
    license="MIT",
    packages=["fm", "fm/downstream", "fm/downstream/pairwise_predictor"],
    data_files=[("source_docs/fm", ["LICENSE", "README.md"])],
    zip_safe=True,
    install_requires = [
        'numpy==1.22.0',
        'pandas==1.3.1',
        'tqdm==4.62',
        'scikit-learn==0.24',
    ],
)