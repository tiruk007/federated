load("@rules_python//python:defs.bzl", "py_binary")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "load_flair",
    srcs = ["load_flair.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)
