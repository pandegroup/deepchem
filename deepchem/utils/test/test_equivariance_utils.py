import unittest
import math
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

if has_torch:
    from deepchem.utils import equivariance_utils


class TestEquivarianceUtils(unittest.TestCase):

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_generators_shape(self) -> None:
        # Test if the output has the correct shape.
        j_values = [1, 2, 3, 5,
                    7]  # Test for multiple quantum angular momentum values
        for j in j_values:
            with self.subTest(j=j):
                generators = equivariance_utils.su2_generators(j)
                self.assertEqual(generators.shape, (3, 2 * j + 1, 2 * j + 1))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_generators_zero_momenta(self) -> None:
        # Test for the case of zero momentum (j=0).
        j = 0
        generators = equivariance_utils.su2_generators(j)
        expected_generators = torch.zeros((3, 1, 1), dtype=torch.complex64)
        self.assertTrue(torch.allclose(generators, expected_generators))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_generators_angular_momentum_one(self) -> None:
        # Test for the case of momentum j=1 (spin-1).
        j = 1
        generators = equivariance_utils.su2_generators(j)
        # Expected J_x, J_z, J_y matrices for j=1

        expected_generators = torch.tensor(
            [[[0.0000 + 0.0000j, 0.7071 + 0.0000j, 0.0000 + 0.0000j],
              [-0.7071 + 0.0000j, 0.0000 + 0.0000j, 0.7071 + 0.0000j],
              [0.0000 + 0.0000j, -0.7071 + 0.0000j, 0.0000 + 0.0000j]],
             [[-0.0000 - 1.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
              [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
              [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 1.0000j]],
             [[0.0000 - 0.0000j, 0.0000 + 0.7071j, 0.0000 - 0.0000j],
              [0.0000 + 0.7071j, 0.0000 - 0.0000j, 0.0000 + 0.7071j],
              [0.0000 - 0.0000j, 0.0000 + 0.7071j, 0.0000 - 0.0000j]]])

        self.assertTrue(torch.allclose(generators, expected_generators))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_commutation(self):
        j_values = [0, 0.5, 1, 1.5, 2,
                    2.5]  # Test for multiple quantum angular momentum values
        for j in j_values:
            with self.subTest(j=j):
                X = equivariance_utils.su2_generators(j)
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[0], X[1]),
                                   X[2]))
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[1], X[2]),
                                   X[0]))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_j_0(self):
        # Test for j = 0, which means we have a 1x1 transformation matrix
        j = 0
        Q = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex128)
        expected_Q = torch.tensor([[1]], dtype=torch.complex128)
        self.assertTrue(torch.allclose(Q, expected_Q))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_j_2(self):
        # Test for j = 2, which means we have a 5x5 transformation matrix
        j = 2
        Q = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex64)
        expected_Q = torch.tensor(
            [[
                0.0000 + 0.7071j, -0.0000 + 0.0000j, -0.0000 + 0.0000j,
                -0.0000 + 0.0000j, -0.7071 + 0.0000j
            ],
             [
                 -0.0000 + 0.0000j, 0.0000 + 0.7071j, -0.0000 + 0.0000j,
                 -0.7071 + 0.0000j, -0.0000 + 0.0000j
             ],
             [
                 -0.0000 + 0.0000j, -0.0000 + 0.0000j, -1.0000 + 0.0000j,
                 -0.0000 + 0.0000j, -0.0000 + 0.0000j
             ],
             [
                 -0.0000 + 0.0000j, 0.0000 + 0.7071j, -0.0000 + 0.0000j,
                 0.7071 - 0.0000j, -0.0000 + 0.0000j
             ],
             [
                 -0.0000 - 0.7071j, -0.0000 + 0.0000j, -0.0000 + 0.0000j,
                 -0.0000 + 0.0000j, -0.7071 + 0.0000j
             ]],
            dtype=torch.complex64)
        self.assertTrue(torch.allclose(Q, expected_Q))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_device(self) -> None:
        # Test for device placement (CPU to CUDA)
        j = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Q = equivariance_utils.change_basis_real_to_complex(j, device=device)
        self.assertEqual(Q.device, device)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_dtype_conversion(self):
        # Test for dtype conversion (complex128 to complex64)
        j = 1
        Q_complex128 = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex128)
        Q_complex64 = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex64)

        self.assertEqual(Q_complex128.dtype, torch.complex128)
        self.assertEqual(Q_complex64.dtype, torch.complex64)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_generators_shape(self):
        j_values = [1, 2, 3, 4, 5]
        for j in j_values:
            with self.subTest(j=j):
                result = equivariance_utils.so3_generators(j)
                expected_shape = (3, 2 * j + 1, 2 * j + 1)
                self.assertEqual(result.shape, expected_shape)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_commutation(self):
        j_values = [0, 1, 2, 3, 4,
                    5]  # Test for multiple quantum angular momentum values
        for j in j_values:
            with self.subTest(j=j):
                X = equivariance_utils.so3_generators(j)
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[0], X[1]),
                                   X[2]))
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[1], X[2]),
                                   X[0]))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_j0(self):
        j = 0
        result = equivariance_utils.so3_generators(j)
        expected = torch.tensor([[[0.]], [[0.]], [[0.]]],
                                dtype=torch.float64).float()
        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_j1(self):
        j = 1
        result = equivariance_utils.so3_generators(j)
        expected = torch.tensor(
            [[[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, -1.0000],
              [0.0000, 1.0000, 0.0000]],
             [[0.0000, 0.0000, 1.0000], [0.0000, 0.0000, 0.0000],
              [-1.0000, 0.0000, 0.0000]],
             [[0.0000, -1.0000, 0.0000], [1.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]]],
            dtype=torch.float64).float()
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_unitary_property(self):
        j = 2
        alpha = torch.tensor([0.2])
        beta = torch.tensor([0.1])
        gamma = torch.tensor([0.7])

        D_matrix = equivariance_utils.wigner_D(j, alpha, beta, gamma)
        D_matrix = D_matrix[0]
        conjugate_transpose = torch.transpose(torch.conj(D_matrix), 0, 1)
        identity_matrix = torch.eye(D_matrix.shape[0], dtype=D_matrix.dtype)

        self.assertTrue(
            torch.allclose(D_matrix @ conjugate_transpose,
                           identity_matrix,
                           atol=1e-5))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_orthogonality(self):
        j = 2
        alpha = torch.tensor([0.2])
        beta = torch.tensor([0.1])
        gamma = torch.tensor([0.7])

        D_matrix = equivariance_utils.wigner_D(j, alpha, beta, gamma)
        D_matrix = D_matrix[0]
        num_columns = D_matrix.shape[1]

        for col1 in range(num_columns):
            for col2 in range(num_columns):
                if col1 != col2:
                    dot_product = torch.dot(D_matrix[:, col1], D_matrix[:,
                                                                        col2])
                    self.assertAlmostEqual(dot_product.item(), 0.0, places=5)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_semifactorial(self) -> None:
        # Test base cases
        self.assertEqual(equivariance_utils.semifactorial(0), 1.0)
        self.assertEqual(equivariance_utils.semifactorial(1), 1.0)

        self.assertEqual(equivariance_utils.semifactorial(5),
                         15.0)  # 5!! = 5 * 3 * 1
        self.assertEqual(equivariance_utils.semifactorial(6),
                         48.0)  # 6!! = 6 * 4 * 2

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_pochhammer_base_case(self) -> None:
        # Test k=0 (should return 1.0)
        self.assertEqual(equivariance_utils.pochhammer(3, 0), 1.0)

        self.assertEqual(equivariance_utils.pochhammer(3, 4),
                         360.0)  # (3)_4 = 3 * 4 * 5 * 6
        self.assertEqual(equivariance_utils.pochhammer(5, 2),
                         30.0)  # (5)_2 = 5 * 6

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_lpmv(self) -> None:
        x = torch.tensor([0.5])
        # Test P_2^1(x)
        result = equivariance_utils.lpmv(2, 1, x)
        expected = torch.tensor([-1.2990])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test P_1^1(x)
        result = equivariance_utils.lpmv(1, 1, x)
        expected = torch.tensor([-math.sqrt(1 - 0.5**2)])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test P_2^2(x)
        result = equivariance_utils.lpmv(2, 2, x)
        expected = torch.tensor([3 * (1 - 0.5**2)])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_harmonics(self) -> None:
        theta = torch.tensor([0.0, math.pi / 2])
        phi = torch.tensor([0.0, math.pi])
        sh = equivariance_utils.SphericalHarmonics()

        # Test Y_1^0(theta, phi)
        result = sh.get_element(1, 0, theta, phi)
        expected = torch.tensor([0.4886, -0.0000])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test Y_1^1(theta, phi)
        result = sh.get_element(1, 1, theta, phi)
        expected = torch.tensor([-0.0000, 0.4886])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

        # Get all spherical harmonics
        result = sh.get(1, theta, phi)
        expected = torch.tensor([[-0.0000, 0.4886, -0.0000],
                                 [0.0000, -0.0000, 0.4886]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_irr_repr(self) -> None:
        # Test irreducible representation of SO3.
        order = 1
        alpha = torch.tensor(0.1)
        beta = torch.tensor(0.2)
        gamma = torch.tensor(0.3)

        # Edge case: order = 0.
        order_zero = 0
        result_zero = equivariance_utils.irr_repr(order_zero, alpha, beta,
                                                  gamma)
        self.assertTrue(torch.allclose(result_zero, torch.tensor([[1.0]])))

        result = equivariance_utils.irr_repr(order, alpha, beta, gamma)
        expected = torch.tensor([[0.9216, 0.0587, 0.3836],
                                 [0.0198, 0.9801, -0.1977],
                                 [-0.3875, 0.1898, 0.9021]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))
