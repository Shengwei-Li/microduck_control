"""Tests for the payload body. They need mujoco and the upstream robot model,
but do not start a simulation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mjlab_microduck")

from payload_duck import payload  # noqa: E402


def upstream_spec():
    import mjlab_microduck.tasks  # noqa: F401  Import order: avoids a circular import upstream.
    from mjlab_microduck.robot.microduck_constants import get_walk_spec

    return get_walk_spec()


@pytest.mark.parametrize("mount", ["mouth", "trunk"])
def test_payload_body_adds_no_mass(mount):
    base = upstream_spec().compile()
    model = payload.add_payload_body(upstream_spec(), mount).compile()
    assert model.nbody == base.nbody + 1
    assert model.body_mass.sum() == pytest.approx(base.body_mass.sum())
    assert model.nq == base.nq and model.nu == base.nu  # No new joint or actuator.
    body = model.body(payload.PAYLOAD_BODY)
    assert model.body(body.parentid[0]).name == payload.MOUNTS[mount].parent
    assert tuple(body.pos) == pytest.approx(payload.MOUNTS[mount].pos)
    geom = model.geom("payload_visual")
    assert geom.contype[0] == 0 and geom.conaffinity[0] == 0  # No contact.


def test_mouth_mount_is_the_upstream_mouth_site():
    import mujoco

    model = payload.add_payload_body(upstream_spec(), "mouth").compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = data.site("mouth_tip").xpos
    assert data.body(payload.PAYLOAD_BODY).xpos == pytest.approx(site, abs=1e-6)
    # The beak is in front of the trunk origin and of the feet.
    assert site[0] - data.body("trunk_base").xpos[0] > 0.03


def test_robot_mass_constant_matches_model():
    assert upstream_spec().compile().body_mass.sum() == pytest.approx(payload.ROBOT_MASS_KG, abs=1e-3)


def test_unit_inertia():
    # Box 30 x 30 x 20 mm, 1 kg: Ixx = (0.03^2 + 0.02^2) / 12, Izz = (0.03^2 + 0.03^2) / 12.
    assert payload.MOUNTS["trunk"].unit_inertia() == pytest.approx([1.0833e-4, 1.0833e-4, 1.5e-4], rel=1e-3)
    # Solid sphere, radius 15 mm, 1 kg: 2/5 * 0.015^2.
    assert payload.MOUNTS["mouth"].unit_inertia() == pytest.approx([9.0e-5] * 3)


def test_com_shift():
    # 240 g at 30 mm forward: 0.24 * 0.03 / (0.737 + 0.24) = 7.37 mm.
    assert payload.com_shift_m(0.24, 0.03) == pytest.approx(0.00737, abs=1e-5)
    assert payload.com_shift_m(0.0, 0.03) == 0.0
