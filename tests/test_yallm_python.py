"""Unit tests for the python wrapper's path-location logic."""

import os


from yallm import (
    YaLLMNotFound,
    _join,
    _matching_parents,
    _module_path,
    _user_scheme,
    find_yallm_bin,
)


def test_matching_parents_matches_unix_suffix():
    path = "/usr/local/lib/python3.13/site-packages/yallm"
    assert _matching_parents(path, "lib/python*/site-packages/yallm") == "/usr/local"


def test_matching_parents_returns_none_for_short_path():
    assert _matching_parents("yallm", "lib/site-packages/yallm") is None
    assert _matching_parents(None, "yallm") is None


def test_matching_parents_returns_none_when_suffix_mismatch():
    assert _matching_parents("/opt/packages/yallm", "lib/site-packages/yallm") is None


def test_join_passthrough_and_none():
    assert _join("/usr/local", "bin") == os.path.join("/usr/local", "bin")
    assert _join(None, "bin") is None


def test_module_path_is_non_empty_dir():
    path = _module_path()
    assert path
    assert os.path.isdir(path)


def test_user_scheme_is_str():
    assert isinstance(_user_scheme(), str)


def test_find_yallm_bin_returns_existing_file_or_raises():
    try:
        found = find_yallm_bin()
    except YaLLMNotFound:
        return  # no installed scripts dir in a source checkout
    assert os.path.isfile(found), f"{found} does not exist"
