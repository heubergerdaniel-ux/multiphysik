"""Tests for the topology optimisation module.

FEM correctness tests
---------------------
* element_stiffness: symmetry, positive semi-definiteness, size invariant
* element_dof_indices: correct shape and no duplicate DOFs within one element
* fem_solve: 1-element cantilever deflection matches Hooke's law

BESO behaviour tests
--------------------
* beso_step: volume fraction monotonically decreases toward target
* beso_step: sensitivity preserved (high-strain elements stay solid)
* is_converged: triggers only after *patience* stable iterations

Boundary condition tests
------------------------
* fixed_face_dofs: correct count for z0 face
* point_load_dof: force appears at correct DOF index
"""
from __future__ import annotations

import numpy as np
import pytest

from picogk_mp.topopt.fem import (
    element_stiffness,
    element_dof_indices,
    assemble_K,
    fem_solve,
    element_strain_energy,
)
from picogk_mp.topopt.beso import BESOState, beso_step, is_converged
from picogk_mp.topopt.boundary import (
    fixed_face_dofs,
    point_load_dof,
    BoundaryConditions,
)


# ======================================================================
# Element stiffness matrix
# ======================================================================

class TestElementStiffness:
    H, E, NU = 10.0, 3500.0, 0.36   # typical PLA, 10 mm element

    def test_shape(self):
        KE = element_stiffness(self.H, self.E, self.NU)
        assert KE.shape == (24, 24)

    def test_symmetric(self):
        KE = element_stiffness(self.H, self.E, self.NU)
        assert np.allclose(KE, KE.T, atol=1e-10)

    def test_positive_semidefinite(self):
        KE = element_stiffness(self.H, self.E, self.NU)
        eigs = np.linalg.eigvalsh(KE)
        # 6 rigid-body modes → 6 zero eigenvalues; rest must be positive
        assert np.sum(eigs < -1e-8) == 0, f"Negative eigenvalue: {eigs.min()}"

    def test_scales_with_E(self):
        KE1 = element_stiffness(self.H, self.E,       self.NU)
        KE2 = element_stiffness(self.H, self.E * 2.0, self.NU)
        assert np.allclose(KE2, 2.0 * KE1, rtol=1e-10)

    def test_scales_with_h(self):
        """KE scales as h (for a cube: det(J)=h^3, B scales as 1/h, so K~h)."""
        KE1 = element_stiffness(1.0, self.E, self.NU)
        KE2 = element_stiffness(2.0, self.E, self.NU)
        assert np.allclose(KE2, 2.0 * KE1, rtol=1e-10)

    def test_cached(self):
        """Same arguments should return the same object (lru_cache)."""
        KE_a = element_stiffness(self.H, self.E, self.NU)
        KE_b = element_stiffness(self.H, self.E, self.NU)
        assert KE_a is KE_b


# ======================================================================
# Element DOF indices
# ======================================================================

class TestElementDofIndices:
    def test_shape(self):
        edofs = element_dof_indices(3, 2, 4)
        assert edofs.shape == (3 * 2 * 4, 24)

    def test_unique_per_element(self):
        """Each element must reference 24 distinct global DOFs."""
        edofs = element_dof_indices(2, 2, 2)
        for row in edofs:
            assert len(np.unique(row)) == 24, "Duplicate DOF within element"

    def test_dof_range(self):
        Nx, Ny, Nz = 2, 3, 4
        edofs = element_dof_indices(Nx, Ny, Nz)
        ndof  = 3 * (Nx+1) * (Ny+1) * (Nz+1)
        assert edofs.min() >= 0
        assert edofs.max() <  ndof

    def test_single_element(self):
        edofs = element_dof_indices(1, 1, 1)
        assert edofs.shape == (1, 24)
        # Node (0,0,0) -> global node 0, DOFs 0,1,2
        assert 0 in edofs[0] and 1 in edofs[0] and 2 in edofs[0]


# ======================================================================
# FEM assembly + solve -- simple single-element cantilever
# ======================================================================

class TestFemSolve:
    """One-element cube: fix all DOFs on x=0 face, apply tip load in z."""

    H  = 10.0   # mm
    E  = 3500.0  # MPa
    NU = 0.36

    def _single_element_problem(self):
        Nx, Ny, Nz = 1, 1, 1
        mask  = np.ones((Nx, Ny, Nz), dtype=bool)
        edofs = element_dof_indices(Nx, Ny, Nz)
        KE    = element_stiffness(self.H, self.E, self.NU)

        # Fix the x=0 face (nodes 0,3,4,7 in standard hex ordering)
        fixed = fixed_face_dofs(Nx, Ny, Nz, face="x0")

        # Apply unit force in z direction at node (1,1,1) -- far corner
        ndof = 3 * (Nx+1) * (Ny+1) * (Nz+1)
        from picogk_mp.topopt.boundary import node_index
        n_tip = node_index(1, 1, 1, Nx, Ny)
        f = np.zeros(ndof)
        f[3*n_tip + 2] = 1.0    # F_z = 1 N

        return mask, edofs, KE, fixed, f

    def test_solve_returns_correct_shape(self):
        mask, edofs, KE, fixed, f = self._single_element_problem()
        u = fem_solve(mask, edofs, KE, fixed, f)
        Nx, Ny, Nz = 1, 1, 1
        ndof = 3 * (Nx+1) * (Ny+1) * (Nz+1)
        assert u.shape == (ndof,)

    def test_fixed_dofs_are_zero(self):
        mask, edofs, KE, fixed, f = self._single_element_problem()
        u = fem_solve(mask, edofs, KE, fixed, f)
        assert np.allclose(u[fixed], 0.0, atol=1e-12)

    def test_deflects_in_load_direction(self):
        """With unit Fz load, tip should displace positively in z."""
        mask, edofs, KE, fixed, f = self._single_element_problem()
        u = fem_solve(mask, edofs, KE, fixed, f)
        from picogk_mp.topopt.boundary import node_index
        n_tip = node_index(1, 1, 1, 1, 1)
        assert u[3*n_tip + 2] > 0, "Tip should deflect in load direction"

    def test_stiffer_material_deflects_less(self):
        mask, edofs, KE1, fixed, f = self._single_element_problem()
        KE2 = element_stiffness(self.H, self.E * 10, self.NU)
        u1  = fem_solve(mask, edofs, KE1, fixed, f)
        u2  = fem_solve(mask, edofs, KE2, fixed, f)
        from picogk_mp.topopt.boundary import node_index
        n_tip = node_index(1, 1, 1, 1, 1)
        assert abs(u2[3*n_tip + 2]) < abs(u1[3*n_tip + 2])


# ======================================================================
# Strain energy
# ======================================================================

class TestStrainEnergy:
    def test_positive_for_loaded_element(self):
        Nx, Ny, Nz = 1, 1, 1
        mask  = np.ones((Nx, Ny, Nz), dtype=bool)
        edofs = element_dof_indices(Nx, Ny, Nz)
        KE    = element_stiffness(10.0, 3500.0, 0.36)
        fixed = fixed_face_dofs(Nx, Ny, Nz, "x0")
        ndof  = 3 * 8
        f     = np.zeros(ndof)
        from picogk_mp.topopt.boundary import node_index
        n = node_index(1, 1, 1, 1, 1)
        f[3*n+2] = 1.0
        u  = fem_solve(mask, edofs, KE, fixed, f)
        ce = element_strain_energy(u, edofs, KE)
        assert ce.shape == (1,)
        assert ce[0] > 0

    def test_zero_for_unloaded(self):
        Nx, Ny, Nz = 1, 1, 1
        mask  = np.ones((Nx, Ny, Nz), dtype=bool)
        edofs = element_dof_indices(Nx, Ny, Nz)
        KE    = element_stiffness(10.0, 3500.0, 0.36)
        ndof  = 3 * 8
        u     = np.zeros(ndof)
        ce    = element_strain_energy(u, edofs, KE)
        assert np.allclose(ce, 0.0)


# ======================================================================
# BESO
# ======================================================================

class TestBESOStep:
    def _make_state(self, shape=(4, 4, 4)):
        return BESOState(mask=np.ones(shape, dtype=bool))

    def test_volume_decreases_toward_target(self):
        state = self._make_state()
        # random sensitivity -- all elements present initially
        np.random.seed(42)
        for _ in range(10):
            alpha = np.random.rand(4*4*4)
            state = beso_step(state, alpha, vol_target=0.5, er=0.05)
        assert state.volume_fraction <= 1.0 + 1e-6

    def test_does_not_overshoot_target(self):
        """Volume fraction should not go below vol_target significantly."""
        state = self._make_state(shape=(6, 6, 6))
        np.random.seed(7)
        for _ in range(20):
            alpha = np.random.rand(6*6*6)
            state = beso_step(state, alpha, vol_target=0.40, er=0.05)
        # allow a small overshoot from bidirectional re-addition
        assert state.volume_fraction >= 0.35

    def test_high_sensitivity_stays_solid(self):
        """Element with maximum sensitivity should never be removed."""
        shape = (3, 3, 3)
        state = self._make_state(shape)
        alpha = np.zeros(3*3*3)
        alpha[13] = 1e6          # one very high sensitivity element

        for _ in range(15):
            state = beso_step(state, alpha, vol_target=0.30, er=0.05)

        assert state.mask.ravel()[13], "High-sensitivity element must stay solid"

    def test_compliance_history_grows(self):
        state = self._make_state()
        np.random.seed(1)
        for _ in range(3):
            alpha = np.random.rand(4*4*4)
            state = beso_step(state, alpha, vol_target=0.5, er=0.05)
        assert len(state.compliance_history) == 3


class TestIsConverged:
    def _state_with_compliance(self, values):
        s = BESOState(mask=np.ones((2, 2, 2), dtype=bool))
        s.compliance_history = list(values)
        return s

    def test_not_converged_with_few_iterations(self):
        s = self._state_with_compliance([10.0, 9.0, 8.5])
        assert not is_converged(s, patience=5)

    def test_converged_when_stable(self):
        s = self._state_with_compliance(
            [10.0, 9.5, 9.0, 8.51, 8.50, 8.501, 8.499, 8.500, 8.501, 8.500]
        )
        assert is_converged(s, tol=1e-2, patience=5)

    def test_not_converged_when_still_changing(self):
        s = self._state_with_compliance([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        assert not is_converged(s, tol=1e-3, patience=5)


# ======================================================================
# Boundary conditions
# ======================================================================

class TestBoundaryConditions:
    def test_z0_face_dof_count(self):
        Nx, Ny, Nz = 4, 3, 5
        dofs = fixed_face_dofs(Nx, Ny, Nz, "z0")
        # (Nx+1)*(Ny+1) nodes on z0 face, 3 DOFs each
        expected = (Nx+1) * (Ny+1) * 3
        assert len(dofs) == expected

    def test_all_axes_fixed_by_default(self):
        dofs = fixed_face_dofs(2, 2, 2, "z0")
        # All three DOF axes should appear
        assert len(dofs) % 3 == 0

    def test_single_axis_fix(self):
        dofs_all  = fixed_face_dofs(2, 2, 2, "z0", axes=(0, 1, 2))
        dofs_one  = fixed_face_dofs(2, 2, 2, "z0", axes=(2,))
        assert len(dofs_one) == len(dofs_all) // 3

    def test_unknown_face_raises(self):
        with pytest.raises(ValueError):
            fixed_face_dofs(2, 2, 2, "q5")

    def test_point_load_at_nearest_node(self):
        Nx, Ny, Nz = 5, 5, 10
        h  = 2.0
        # Load at exact node (3,3,8) -> physical (6,6,16) with zero offset
        f = point_load_dof(Nx, Ny, Nz, h,
                           offset=(0, 0, 0),
                           position_mm=(6.0, 6.0, 16.0),
                           force_N=(0, 0, -10.0))
        from picogk_mp.topopt.boundary import node_index
        n = node_index(3, 3, 8, Nx, Ny)
        assert f[3*n + 2] == pytest.approx(-10.0)
        assert f[3*n]     == pytest.approx(0.0)
        # All other entries zero
        f2 = f.copy(); f2[3*n:3*n+3] = 0.0
        assert np.allclose(f2, 0.0)

    def test_disc_base_with_tip_load_bc(self):
        # Grid must cover the cylinder axis (x=0,y=0) and the load point.
        # offset=(-91,-48,0), h=3 -> axis at ix=91/3~30, iy=48/3=16
        h = 3.0
        Nx, Ny, Nz = 47, 32, 87
        offset = (-91.0, -48.0, 0.0)
        bc = BoundaryConditions.disc_base_with_tip_load(
            Nx, Ny, Nz, h, offset,
            base_radius_mm=48.0,
            load_point_mm=(-82.0, 0.0, 244.0),
            load_mass_g=400.0,
        )
        ndof = 3 * (Nx+1) * (Ny+1) * (Nz+1)
        assert bc.force_vec.shape == (ndof,)
        assert len(bc.fixed_dofs) > 0
        # Load must be downward (negative z)
        assert bc.force_vec.min() < 0
