import setuptools

setuptools.setup(
    name="sccf",
    version="0.0.1",
    author="Shane Barratt",
    description="Minimizing a sum of clipped convex functions",
    license="XXX",
    url="XXX",
    packages=setuptools.find_packages(),
    install_requires=[
        "cvxpy >= 1.0",
        "numpy >= 1.8",
    ]
)