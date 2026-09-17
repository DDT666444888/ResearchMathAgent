"""Research Math Agent web app: a Claude-powered agent that solves the
*First Proof* benchmark problems, with a live step-by-step UI."""

import site, sys
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

def __getattr__(name):
    # Persistence/locking are also used by the dependency-light CLI. Import the
    # optional Anthropic-backed agent only when its public API is requested.
    if name in {"AgentConfig", "run_agent"}:
        from . import agent
        return getattr(agent, name)
    raise AttributeError(name)

__all__ = ["AgentConfig", "run_agent"]
