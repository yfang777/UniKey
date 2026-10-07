__all__ = ["Executor", "StableExecutor", "ReKepEnv"]


def __getattr__(name):
    if name == "StableExecutor":
        from .executor_stable import StableExecutor

        return StableExecutor
    if name == "Executor":
        from .executor import Executor

        return Executor
    if name == "ReKepEnv":
        from .env import ReKepEnv

        return ReKepEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
