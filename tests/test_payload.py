"""Tests for the payload body. They need mujoco and the upstream robot model,
but do not start a simulation."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mjlab_microduck")

from payload_duck import payload  # noqa: E402


def upstream_spec():
    import mjlab_microduck.tasks  # noqa: F401  Import order: avoids a circular import upstream.
    from mjlab_microduck.robot.microduck_constants import get_walk_spec

    return get_walk_spec()


def test_payload_body_adds_no_mass():
    base = upstream_spec().compile()
    model = payload.add_payload_body(upstream_spec()).compile()
    assert model.nbody == base.nbody + 1
    assert model.body_mass.sum() == pytest.approx(base.body_mass.sum())
    assert model.nq == base.nq and model.nu == base.nu  # No new joint or actuator.
    body = model.body(payload.PAYLOAD_BODY)
    assert model.body(body.parentid[0]).name == "trunk_base"
    assert tuple(body.pos) == pytest.approx(payload.MOUNT_POS)
    geom = model.geom("payload_visual")
    assert geom.contype[0] == 0 and geom.conaffinity[0] == 0  # No contact.


def test_robot_mass_constant_matches_model():
    assert upstream_spec().compile().body_mass.sum() == pytest.approx(payload.ROBOT_MASS_KG, abs=1e-3)


def test_box_inertia():
    # 0.12 kg box, 30 x 30 x 20 mm: Ixx = m/12 * (0.03^2 + 0.02^2), Izz = m/12 * (0.03^2 + 0.03^2).
    inertia = payload.box_inertia(torch.tensor([0.12, 0.0]))
    assert inertia.shape == (2, 3)
    assert inertia[0].tolist() == pytest.approx([1.3e-5, 1.3e-5, 1.8e-5])
    assert inertia[1].tolist() == [0.0, 0.0, 0.0]


def test_com_shift():
    # 240 g at 30 mm forward: 0.24 * 0.03 / (0.737 + 0.24) = 7.37 mm.
    assert payload.com_shift_m(0.24, 0.03) == pytest.approx(0.00737, abs=1e-5)
    assert payload.com_shift_m(0.0, 0.03) == 0.0
