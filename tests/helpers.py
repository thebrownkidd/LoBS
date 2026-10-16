from lobs.backends.fake import FakeBackend


def fake_sim(world, motion, contacts=None, **kw):
    """Simulate a world on the FakeBackend with analytic motion(t) -> dict of arrays."""
    return world.simulate(
        lambda spec, reg, opts: FakeBackend(spec, reg, opts, motion=motion, contacts=contacts), **kw
    )
