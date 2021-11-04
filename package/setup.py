#!/usr/bin/env python

from setuptools import setup


def setup_package():
    setup(
        name="DVD",
        version="1.0",
        description="Dynamic Video Depth",
        packages=[
            "dvd.configs",
            "dvd.datasets",
            "dvd.loggers",
            "dvd.losses",
            "dvd.models",
            "dvd.networks",
            "dvd.options",
            "dvd.scripts",
            "dvd.third_party",
            "dvd.util",
            "dvd.visualize",
            "dvd",
        ],
    )


if __name__ == "__main__":
    setup_package()
