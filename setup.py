from setuptools import setup, find_packages

setup(
    name="semanticapi",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "httpx>=0.26.0",
        "python-dotenv>=1.0.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.5.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.40.0"],
        "openai": ["openai>=1.0.0"],
        "all": ["anthropic>=0.40.0", "openai>=1.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "semanticapi=semanticapi.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["providers/*.json"],
    },
    author="Semantic API Contributors",
    description="Natural language interface to any API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/petermtj/semanticapi-engine",
    license="AGPL-3.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
