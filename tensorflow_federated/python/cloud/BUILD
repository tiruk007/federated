load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":cloud_packages"],
)

package_group(
    name = "cloud_packages",
    packages = [
        "//tensorflow_federated/python/cloud/...",
    ],
)

licenses(["notice"])

py_library(
    name = "cloud",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
)
