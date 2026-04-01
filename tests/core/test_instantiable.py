"""Tests for Instantiable base class."""

import pytest
from pydantic import BaseModel

from ThreeWToolkit.core import Instantiable


class TestInstantiable:
    """Test Instantiable pattern."""

    def test_basic_instantiation(self):
        """Test basic build pattern with Instantiable."""

        class MyClass:
            def __init__(self, config):
                self.value = config.value

        class MyConfig(BaseModel, Instantiable):
            value: int
            target_: type = MyClass

        config = MyConfig(value=42)
        instance = config.build()

        assert isinstance(instance, MyClass)
        assert instance.value == 42

    def test_build_with_extra_args(self):
        """Test build pattern passing extra arguments."""

        class MyClass:
            def __init__(self, config, extra_arg):
                self.value = config.value
                self.extra = extra_arg

        class MyConfig(BaseModel, Instantiable):
            value: str
            target_: type = MyClass

        config = MyConfig(value="test")
        instance = config.build("extra_value")

        assert instance.value == "test"
        assert instance.extra == "extra_value"

    def test_build_with_kwargs(self):
        """Test build pattern passing keyword arguments."""

        class MyClass:
            def __init__(self, config, *, name="default"):
                self.config_value = config.config_value
                self.name = name

        class MyConfig(BaseModel, Instantiable):
            config_value: float
            target_: type = MyClass

        config = MyConfig(config_value=3.14)
        instance = config.build(name="custom")

        assert instance.config_value == 3.14
        assert instance.name == "custom"

    def test_config_accessible_in_built_instance(self):
        """Test that config is accessible in built instance."""

        class MyClass:
            def __init__(self, config):
                self.config = config

        class MyConfig(BaseModel, Instantiable):
            param_a: int
            param_b: str
            target_: type = MyClass

        config = MyConfig(param_a=10, param_b="hello")
        instance = config.build()

        assert instance.config.param_a == 10
        assert instance.config.param_b == "hello"

    def test_nested_config_instantiation(self):
        """Test instantiation with nested configs."""

        class InnerClass:
            def __init__(self, config):
                self.inner_value = config.inner_value

        class InnerConfig(BaseModel, Instantiable):
            inner_value: int
            target_: type = InnerClass

        class OuterClass:
            def __init__(self, config):
                self.outer_value = config.outer_value
                self.inner = config.inner_config.build()

        class OuterConfig(BaseModel, Instantiable):
            outer_value: str
            inner_config: InnerConfig
            target_: type = OuterClass

        inner_config = InnerConfig(inner_value=99)
        outer_config = OuterConfig(outer_value="outer", inner_config=inner_config)
        instance = outer_config.build()

        assert instance.outer_value == "outer"
        assert instance.inner.inner_value == 99
