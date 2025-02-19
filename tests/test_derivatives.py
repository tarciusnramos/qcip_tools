import random
import math
import itertools
import numpy

from qcip_tools import derivatives, derivatives_e, derivatives_g, atom as qcip_atom, \
    molecule as qcip_molecule, numerical_differentiation
from tests import QcipToolsTestCase, factories
from qcip_tools import transformations


def tensor_rotate(tensor, psi, theta, chi):
    """Return a rotated tensor

    .. warning::

        To much magic here, will be deprecated at some point.

    :param tensor: the tensor to be rotated
    :type tensor: numpy.ndarray
    :param psi: Rotation around the Z axis (in degree)
    :type psi: float
    :param theta: rotation around the y' axis (in degree)
    :type theta: float
    :param chi: Rotation around the z'' axis (in degree)
    :type chi: float
    :rtype: numpy.ndarray
    """

    new_tensor = numpy.zeros(tensor.shape)
    order = len(tensor.shape)

    rotation_matrix = transformations.TransformationMatrix.rotate_euler(psi, theta, chi, in_degree=True)[:3, :3]

    for i in itertools.product(range(3), repeat=order):
        tmp = .0
        for j in itertools.product(range(3), repeat=order):
            product = 1
            for k in range(order):
                product *= rotation_matrix[i[k], j[k]]
            tmp += product * tensor[j]

        new_tensor[i] = tmp

    return new_tensor


class DerivativesTestCase(QcipToolsTestCase):

    def setUp(self):
        pass

    def test_derivatives(self):
        """Test the behavior of the Derivate object"""

        # create energy
        e = derivatives.Derivative()
        self.assertEqual(e.diff_representation, '')
        self.assertEqual(e.representation(), '')
        self.assertEqual(e.dimension(), 1)
        self.assertEqual(e.shape(), [1])
        self.assertIsNone(e.basis)
        self.assertEqual(e.order(), 0)

        # test smart iterator (yes, also with energy which is basically a scalar)
        num_smart_iterator_call = 0

        r = numpy.zeros(e.shape()).flatten()
        for i in e.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < e.dimension(), i)
            for j in e.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertTrue(numpy.all(r == 1))
        self.assertEqual(num_smart_iterator_call, 1)

        # create derivative
        d0 = derivatives.Derivative(from_representation='FF')
        self.assertEqual(d0.diff_representation, 'FF')
        self.assertEqual(d0.representation(), 'FF')
        self.assertEqual(d0.dimension(), 3 * 3)
        self.assertEqual(d0.shape(), [3, 3])
        self.assertIsNone(d0.basis)
        self.assertEqual(d0.order(), 2)

        # test smart iterator
        num_smart_iterator_call = 0

        r = numpy.zeros(d0.shape()).flatten()
        for i in d0.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < d0.dimension(), i)
            for j in d0.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertTrue(numpy.all(r == 1))
        self.assertEqual(num_smart_iterator_call, 6)  # Note: 6 = 3 * (3+1) / 2

        # differentiate
        d1 = derivatives.Derivative(
            from_representation='G', basis=d0, spacial_dof=9)

        self.assertEqual(d1.diff_representation, 'G')
        self.assertEqual(d1.basis.diff_representation, 'FF')
        self.assertEqual(d1.representation(), 'GFF')

        self.assertEqual(d1.dimension(), 9 * 3 * 3)
        self.assertEqual(d1.shape(), [9, 3, 3])
        self.assertEqual(d1.order(), 3)

        # test smart iterator
        num_smart_iterator_call = 0

        r = numpy.zeros(d1.shape()).flatten()
        for i in d1.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < d1.dimension(), i)
            for j in d1.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertTrue(numpy.all(r == 1))
        self.assertEqual(num_smart_iterator_call, 9 * 6)

        # differentiate again:
        d2 = d1.differentiate('G')
        self.assertEqual(d2.representation(), 'GGFF')
        self.assertEqual(d2.basis.representation(), d1.representation())
        self.assertEqual(d2.order(), 4)

        # test smart iterator
        num_smart_iterator_call = 0

        r = numpy.zeros(d2.shape()).flatten()
        for i in d2.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < d2.dimension(), i)
            for j in d2.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertTrue(numpy.all(r == 1))
        self.assertEqual(num_smart_iterator_call, 6 * 45)  # Note: 45 = 9 * (9+1) / 2

        # tricky one:
        d4 = derivatives.Derivative(from_representation='dDF')
        self.assertEqual(d4.diff_representation, 'dDF')
        self.assertEqual(d4.representation(), 'dDF')
        self.assertEqual(d4.dimension(), 3 * 3 * 3)
        self.assertEqual(d4.shape(), [3, 3, 3])
        self.assertIsNone(d4.basis)
        self.assertEqual(d4.order(), 3)

        # test smart iterator
        num_smart_iterator_call = 0

        r = numpy.zeros(d4.shape()).flatten()
        for i in d4.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < d4.dimension(), i)
            for j in d4.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertEqual(num_smart_iterator_call, 27)
        self.assertTrue(numpy.all(r == 1))

        # another tricky one:
        d5 = derivatives.Derivative(from_representation='XDDF')
        self.assertEqual(d5.diff_representation, 'XDDF')
        self.assertEqual(d5.representation(), 'XDDF')
        self.assertEqual(d5.dimension(), 3 * 3 * 3 * 3)
        self.assertEqual(d5.shape(), [3, 3, 3, 3])
        self.assertIsNone(d5.basis)
        self.assertEqual(d5.order(), 4)

        # test smart iterator
        num_smart_iterator_call = 0

        r = numpy.zeros(d5.shape()).flatten()
        for i in d5.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < d5.dimension(), i)
            for j in d5.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertEqual(num_smart_iterator_call, 54)  # 9 * 6
        self.assertTrue(numpy.all(r == 1))

        # once again, but with full components (not flatten indices)
        num_smart_iterator_call = 0

        r = numpy.zeros(d5.shape())
        for i in d5.smart_iterator(as_flatten=False):
            num_smart_iterator_call += 1
            self.assertEqual(len(i), d5.order())
            self.assertTrue(all(x < y for x, y in zip(i, d5.shape())))
            for j in d5.inverse_smart_iterator(i, as_flatten=False):
                r[j] += 1

        self.assertEqual(num_smart_iterator_call, 54)  # 9 * 6
        self.assertTrue(numpy.all(r.flatten() == 1))

        # test excitations
        d6 = derivatives.Derivative(from_representation='!FF', nstates=10)
        self.assertEqual(d6.diff_representation, '!FF')
        self.assertEqual(d6.representation(), '!FF')
        self.assertEqual(d6.dimension(), 10 * 3 * 3)
        self.assertEqual(d6.shape(), [10, 3, 3])
        self.assertIsNone(d6.basis)
        self.assertEqual(d6.order(), 3)

        # test smart iterator
        num_smart_iterator_call = 0

        r = numpy.zeros(d6.shape()).flatten()
        for i in d6.smart_iterator(as_flatten=True):
            num_smart_iterator_call += 1
            self.assertTrue(i < d6.dimension(), i)
            for j in d6.inverse_smart_iterator(i, as_flatten=True):
                r[j] += 1

        self.assertEqual(num_smart_iterator_call, 10 * 6)
        self.assertTrue(numpy.all(r == 1))

        # make the code cry:
        with self.assertRaises(derivatives.RepresentationError):
            derivatives.Derivative(from_representation='NE', spacial_dof=6)  # "E"
        with self.assertRaises(Exception):
            derivatives.Derivative(from_representation='N')  # missing dof
        with self.assertRaises(Exception):
            derivatives.Derivative(from_representation='G')  # missing dof
        with self.assertRaises(Exception):
            derivatives.Derivative(from_representation='!F')  # missing nstate
        with self.assertRaises(Exception):
            derivatives.Derivative(from_representation='!!F', nstates=10)  # double transition
        with self.assertRaises(Exception):
            derivatives.Derivative(from_representation='##F', nstates=10)  # double excitation
        with self.assertRaises(ValueError):
            d0.differentiate('')  # nothing
        with self.assertRaises(derivatives.RepresentationError):
            d0.differentiate('E')  # "E"
        with self.assertRaises(Exception):
            d0.differentiate('N')  # missing dof
        with self.assertRaises(Exception):
            d0.differentiate('G')  # missing dof

        # test comparison
        self.assertTrue(d1 == 'GFF')
        self.assertTrue(d1 == 'FGF')  # no matter the order
        self.assertTrue(d1 != 'dGD')  # ... But the type matter
        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)

        self.assertTrue(d6 == derivatives.Derivative('F!F', nstates=10))

    def test_tensor(self):
        """Test the behavior of the Tensor object"""

        beta_tensor = numpy.array([a * (-1) ** a for a in range(27)]).reshape((3, 3, 3))
        # Note: a tensor is suppose to be symmetric, which is not the case here
        t = derivatives.Tensor('XDD', components=beta_tensor, frequency='static')

        self.assertEqual(t.representation.representation(), 'XDD')
        self.assertEqual(t.representation.dimension(), 27)
        self.assertEqual(t.frequency, 'static')
        self.assertIsNone(t.spacial_dof)
        self.assertTrue(numpy.array_equal(beta_tensor, t.components))

        gamma_tensor = numpy.array([a * (-1) ** a for a in range(81)]).reshape((3, 3, 3, 3))
        # Note: a tensor is suppose to be symmetric, which is not the case here
        t = derivatives.Tensor('ddDD', components=gamma_tensor, frequency='static')

        self.assertEqual(t.representation.representation(), 'ddDD')
        self.assertEqual(t.representation.dimension(), 81)
        self.assertEqual(t.frequency, 'static')
        self.assertIsNone(t.spacial_dof)
        self.assertTrue(numpy.array_equal(gamma_tensor, t.components))

        # make the code cry:
        with self.assertRaises(ValueError):
            derivatives.Tensor('XDD')  # no frequency
        with self.assertRaises(Exception):
            derivatives.Tensor('N')  # no dof

    def test_sum(self):
        """Just test that the explicit sum and the shorthand gives the same results
        """

        beta_tensor = factories.FakeFirstHyperpolarizabilityTensor(octupolar_factor=2)

        # compute explicit J=3 contribution of beta tensor
        tmp = .0
        for i in derivatives.COORDINATES_LIST:
            for j in derivatives.COORDINATES_LIST:
                for k in derivatives.COORDINATES_LIST:
                    tmp -= 1 / 15 * beta_tensor.components[i, j, j] * beta_tensor.components[i, k, k]
                    tmp -= 4 / 15 * beta_tensor.components[i, i, j] * beta_tensor.components[j, k, k]
                    tmp -= 4 / 15 * beta_tensor.components[i, i, j] * beta_tensor.components[k, j, k]
                    tmp += 5 / 15 * beta_tensor.components[i, j, k] ** 2
                    tmp += 10 / 15 * beta_tensor.components[i, j, k] * beta_tensor.components[j, i, k]

        # now, with shorthand:
        self.assertEqual(tmp, 1 / 15 * beta_tensor.compute_sum([
            (5, 'ijkijk'),
            (10, 'ijkjik'),
            (-1, 'ijjikk'),
            (-4, 'iijkjk'),
            (-4, 'iijjkk')
        ]))

    def test_electrical_derivatives_tensors(self):
        """Test the objects in derivatives_e.py

        Also test that the properties that are invariant under rotation remain invariant !
        """

        angles_set = [
            (0, 0, 0),  # first test without any rotation
            (180, 60, -60),
            (15, -15, 25),
            (450, -75, 75),
            # and the last one, to be certain
            (random.randrange(-360, 360), random.randrange(-360, 360), random.randrange(-360, 360))
        ]

        # static water, HF/d-aug-cc-pVDZ (Gaussian)
        dipole = numpy.array([.0, .0, -0.767392])

        d = derivatives_e.ElectricDipole(dipole=dipole)
        self.assertAlmostEqual(d.norm(), 0.767392)

        alpha = numpy.array(
            [[0.777020e1, .0, .0],
             [.0, 0.895381e1, .0],
             [.0, .0, 0.832281e1]]
        )

        for angles in angles_set:  # invariants remain invariants under rotation
            new_alpha = tensor_rotate(alpha, *angles)
            na = derivatives_e.PolarisabilityTensor(tensor=new_alpha)

            self.assertAlmostEqual(na.isotropic_value(), 0.834894e1, places=3)
            self.assertAlmostEqual(na.anisotropic_value(), 0.102578e1, places=3)

        beta = numpy.array(
            [[[.0, .0, -0.489173],
              [.0, .0, .0],
              [-0.489173, .0, .0]],
             [[.0, .0, .0],
              [.0, .0, 9.080568],
              [.0, 9.080568, .0]],
             [[-0.489173, .0, .0],
              [.0, 9.080568, .0],
              [.0, .0, 4.276656]]]
        )

        b = derivatives_e.FirstHyperpolarisabilityTensor(tensor=beta)

        self.assertAlmostEqual(b.beta_parallel(dipole), -7.7208, places=3)
        self.assertAlmostEqual(b.beta_perpendicular(dipole), -2.5736, places=3)
        self.assertAlmostEqual(b.beta_kerr(dipole), -7.7208, places=3)
        self.assertArraysAlmostEqual(b.beta_vector(), [.0, .0, 12.8681])

        # NOTE: above properties are also invariant to rotation, but in a less funny way.

        for angles in angles_set:
            new_beta = tensor_rotate(beta, *angles)
            nb = derivatives_e.FirstHyperpolarisabilityTensor(tensor=new_beta)

            self.assertAlmostEqual(nb.beta_squared_zzz(), 29.4147, places=3)
            self.assertAlmostEqual(nb.beta_squared_zxx(), 8.5707, places=3)
            self.assertAlmostEqual(nb.beta_hrs(), 6.1632, places=3)
            self.assertAlmostEqual(nb.depolarization_ratio(), 3.4320, places=3)
            self.assertAlmostEqual(nb.octupolar_contribution_squared(), 167.0257, places=3)
            self.assertAlmostEqual(nb.dipolar_contribution_squared(), 99.3520, places=3)
            self.assertAlmostEqual(nb.nonlinear_anisotropy(), 1.2966, places=3)

            self.assertAlmostEqual(nb.polarization_angle_dependant_intensity(0), nb.beta_squared_zxx(), places=4)
            self.assertAlmostEqual(nb.polarization_angle_dependant_intensity(90), nb.beta_squared_zzz(), places=4)

            # check that new and old version are equal
            self.assertAlmostEqual(
                nb.octupolar_contribution_squared(old_version=True),
                nb.octupolar_contribution_squared(old_version=False),
                places=3)
            self.assertAlmostEqual(
                nb.dipolar_contribution_squared(old_version=True),
                nb.dipolar_contribution_squared(old_version=False),
                places=3)

        # static NH3, HF/d-aug-cc-pVDZ (Gaussian)
        dipole = derivatives_e.ElectricDipole(dipole=[.0, .0, 0.625899])

        d = derivatives_e.ElectricDipole(dipole=dipole.components)
        self.assertAlmostEqual(d.norm(), 0.625899)

        alpha = numpy.array(
            [[0.125681e2, .0, -0.485486e-4],
             [.0, 0.125681e2, .0],
             [-0.485486e-4, .0, 0.132024e2]]
        )

        for angles in angles_set:
            new_alpha = tensor_rotate(alpha, *angles)
            na = derivatives_e.PolarisabilityTensor(tensor=new_alpha)

            self.assertAlmostEqual(na.isotropic_value(), 0.127795e2, places=3)
            self.assertAlmostEqual(na.anisotropic_value(), 0.634350, places=3)

        beta = numpy.array(
            [[[9.258607, -0.012368, -6.097955],
              [-0.012368, -9.257993, .0],
              [-6.097955, .0, -0.000073]],
             [[-0.012368, -9.257993, .0],
              [-9.257993, 0.012368, -6.097633],
              [.0, -6.097633, .0]],
             [[-6.097955, .0, -0.000073],
              [.0, -6.097633, .0],
              [-0.000073, .0, -6.483421]]]
        )

        b = derivatives_e.FirstHyperpolarisabilityTensor(tensor=beta)

        self.assertAlmostEqual(b.beta_parallel(dipole), -11.2074, places=3)
        self.assertAlmostEqual(b.beta_perpendicular(dipole), -3.7358, places=3)
        self.assertAlmostEqual(b.beta_kerr(dipole), -11.2074, places=3)
        self.assertArraysAlmostEqual(b.beta_vector(), [.0, .0, -18.6790])

        for angles in angles_set:
            new_beta = tensor_rotate(beta, *angles)
            nb = derivatives_e.FirstHyperpolarisabilityTensor(tensor=new_beta)

            self.assertAlmostEqual(nb.beta_squared_zzz(), 64.6483, places=3)
            self.assertAlmostEqual(nb.beta_squared_zxx(), 19.8385, places=3)
            self.assertAlmostEqual(nb.beta_hrs(), 9.1917, places=3)
            self.assertAlmostEqual(nb.depolarization_ratio(), 3.2587, places=3)

            self.assertAlmostEqual(nb.octupolar_contribution_squared(), 398.6438, places=3)
            self.assertAlmostEqual(nb.dipolar_contribution_squared(), 209.3432, places=3)
            self.assertAlmostEqual(nb.nonlinear_anisotropy(), 1.3799, places=3)
            self.assertAlmostEqual(nb.spherical_J2_contribution_squared(), .0, places=3)

            # check that new and old version are equal
            self.assertAlmostEqual(
                nb.octupolar_contribution_squared(old_version=True),
                nb.octupolar_contribution_squared(old_version=False),
                places=3)
            self.assertAlmostEqual(
                nb.dipolar_contribution_squared(old_version=True),
                nb.dipolar_contribution_squared(old_version=False),
                places=3)

        # static CH4, HF/d-aug-cc-pVDZ (Gaussian)
        alpha = numpy.array(
            [[0.159960e2, .0, .0],
             [.0, 0.159960e2, .0],
             [.0, .0, 0.159960e2]]
        )

        for angles in angles_set:
            new_alpha = tensor_rotate(alpha, *angles)
            na = derivatives_e.PolarisabilityTensor(tensor=new_alpha)

            self.assertAlmostEqual(na.isotropic_value(), 0.159960e2, places=3)
            self.assertAlmostEqual(na.anisotropic_value(), .0, places=3)

        beta = numpy.array(
            [[[.0, .0, .0],
              [.0, .0, -11.757505],
              [.0, -11.757505, .0]],
             [[.0, .0, -11.757505],
              [.0, .0, .0],
              [-11.757505, .0, .0]],
             [[.0, -11.757505, .0],
              [-11.757505, .0, .0],
              [.0, .0, .0]]]
        )

        for angles in angles_set:
            new_beta = tensor_rotate(beta, *angles)
            nb = derivatives_e.FirstHyperpolarisabilityTensor(tensor=new_beta)

            self.assertAlmostEqual(nb.beta_squared_zzz(), 47.3962, places=3)
            self.assertAlmostEqual(nb.beta_squared_zxx(), 31.5975, places=3)
            self.assertAlmostEqual(nb.beta_hrs(), 8.8878, places=3)
            self.assertAlmostEqual(nb.depolarization_ratio(), 1.5, places=3)

            self.assertAlmostEqual(nb.octupolar_contribution_squared(), 829.4336, places=3)
            self.assertAlmostEqual(nb.dipolar_contribution_squared(), .0, places=3)
            self.assertAlmostEqual(nb.spherical_J2_contribution_squared(), .0, places=3)

            # check that new and old version are equal
            self.assertAlmostEqual(
                nb.octupolar_contribution_squared(old_version=True),
                nb.octupolar_contribution_squared(old_version=False),
                places=3)
            self.assertAlmostEqual(
                nb.dipolar_contribution_squared(old_version=True),
                nb.dipolar_contribution_squared(old_version=False),
                places=3)

        # ... since CH4 has no dipole moment, the rest of the properties failed ;)

        # dynamic (911.3nm) water BLYP/d-aug-cc-pVTZ (Dalton)
        beta = numpy.array(
            [[[0.000000e+00, 0.000000e+00, -1.164262e+01],
              [0.000000e+00, 0.000000e+00, 0.000000e+00],
              [-1.164262e+01, 0.000000e+00, 0.000000e+00]],
             [[0.000000e+00, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -1.451377e+01],
              [0.000000e+00, -1.451377e+01, 0.000000e+00]],
             [[-9.491719e+00, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, -1.475714e+01, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -2.222017e+01]]]
        )

        for angles in angles_set:
            new_beta = tensor_rotate(beta, *angles)
            nb = derivatives_e.FirstHyperpolarisabilityTensor(tensor=new_beta, frequency='911.3nm')

            # Values obtained directly from the contribution matrices
            self.assertAlmostEqual(nb.octupolar_contribution_squared(old_version=False), 123.3727, places=3)
            self.assertAlmostEqual(nb.dipolar_contribution_squared(old_version=False), 1276.4387, places=3)
            self.assertAlmostEqual(nb.spherical_J2_contribution_squared(), 1.9108, places=3)  # see ? There is a J=2 !

            self.assertAlmostEqual(nb.beta_squared_zzz(), 280.551, places=3)
            self.assertAlmostEqual(nb.beta_squared_zxx(), 31.304, places=3)
            self.assertAlmostEqual(nb.beta_hrs(), 17.659, places=3)
            self.assertAlmostEqual(nb.depolarization_ratio(), 8.962, places=3)

            # check that new and old version are NOT (!) equal
            self.assertNotEqual(
                nb.octupolar_contribution_squared(old_version=True),
                nb.octupolar_contribution_squared(old_version=False))
            self.assertNotEqual(
                nb.dipolar_contribution_squared(old_version=True),
                nb.dipolar_contribution_squared(old_version=False))

        # static CH2Cl2, CCS/d-aug-cc-pVDZ (dalton)
        gamma = numpy.array(
            [[[[-4.959904e+03, 4.730379e-03, 0.000000e+00],
               [4.730379e-03, -1.790611e+03, -1.524347e-03],
               [0.000000e+00, -1.524347e-03, -2.052200e+03]],
              [[4.730379e-03, -1.790611e+03, -1.524347e-03],
               [-1.790611e+03, 2.387447e-03, 0.000000e+00],
               [-1.524347e-03, 0.000000e+00, 3.412659e-03]],
              [[0.000000e+00, -1.524347e-03, -2.052200e+03],
               [-1.524347e-03, 0.000000e+00, 3.412659e-03],
               [-2.052200e+03, 3.412659e-03, 0.000000e+00]]],
             [[[4.730379e-03, -1.790611e+03, -1.524347e-03],
               [-1.790611e+03, 2.387447e-03, 0.000000e+00],
               [-1.524347e-03, 0.000000e+00, 3.412659e-03]],
              [[-1.790611e+03, 2.387447e-03, 0.000000e+00],
               [2.387447e-03, -5.193209e+03, -6.751678e-04],
               [0.000000e+00, -6.751678e-04, -2.207921e+03]],
              [[-1.524347e-03, 0.000000e+00, 3.412659e-03],
               [0.000000e+00, -6.751678e-04, -2.207921e+03],
               [3.412659e-03, -2.207921e+03, -2.799551e-03]]],
             [[[0.000000e+00, -1.524347e-03, -2.052200e+03],
               [-1.524347e-03, 0.000000e+00, 3.412659e-03],
               [-2.052200e+03, 3.412659e-03, 0.000000e+00]],
              [[-1.524347e-03, 0.000000e+00, 3.412659e-03],
               [0.000000e+00, -6.751678e-04, -2.207921e+03],
               [3.412659e-03, -2.207921e+03, -2.799551e-03]],
              [[-2.052200e+03, 3.412659e-03, 0.000000e+00],
               [3.412659e-03, -2.207921e+03, -2.799551e-03],
               [0.000000e+00, -2.799551e-03, -9.412690e+03]]]]
        )

        orig_g = derivatives_e.SecondHyperpolarizabilityTensor(tensor=gamma)

        for angles in angles_set:
            new_gamma = tensor_rotate(gamma, *angles)
            ng = derivatives_e.SecondHyperpolarizabilityTensor(tensor=new_gamma)

            self.assertAlmostEqual(ng.gamma_parallel(), orig_g.gamma_parallel(), places=2)
            self.assertAlmostEqual(ng.gamma_perpendicular(), orig_g.gamma_perpendicular(), places=2)
            self.assertAlmostEqual(ng.gamma_kerr(), orig_g.gamma_kerr(), places=2)

            self.assertAlmostEqual(ng.gamma_squared_zzzz(), orig_g.gamma_squared_zzzz(), delta=10)
            self.assertAlmostEqual(ng.gamma_squared_zxxx(), orig_g.gamma_squared_zxxx(), delta=10)
            self.assertAlmostEqual(ng.gamma_ths(), orig_g.gamma_ths(), places=2)
            self.assertAlmostEqual(ng.depolarization_ratio(), orig_g.depolarization_ratio(), places=2)

            self.assertAlmostEqual(
                ng.isotropic_contribution_squared(old_version=True),
                ng.isotropic_contribution_squared(old_version=False),
                places=4
            )

            self.assertAlmostEqual(
                ng.quadrupolar_contribution_squared(old_version=True),
                ng.quadrupolar_contribution_squared(old_version=False),
                places=4
            )

            self.assertAlmostEqual(
                ng.hexadecapolar_contribution_squared(old_version=True),
                ng.hexadecapolar_contribution_squared(old_version=False),
                places=4
            )

            self.assertAlmostEqual(ng.spherical_J1_contribution_squared(), .0, places=2)
            self.assertAlmostEqual(ng.spherical_J3_contribution_squared(), .0, places=2)

        # static CH4, CCS/d-aug-cc-pVDZ (dalton)
        gamma = numpy.array([
            [[[-2.229953e+03, -3.839926e-05, -3.112595e-05],
              [-3.839926e-05, -8.607108e+02, 1.515257e-05],
              [-3.112595e-05, 1.515257e-05, -8.607108e+02]],
             [[-3.839926e-05, -8.607108e+02, 1.515257e-05],
              [-8.607108e+02, -2.006040e-05, -1.040868e-05],
              [1.515257e-05, -1.040868e-05, -1.487595e-05]],
             [[-3.112595e-05, 1.515257e-05, -8.607108e+02],
              [1.515257e-05, -1.040868e-05, -1.487595e-05],
              [-8.607108e+02, -1.487595e-05, -5.166972e-05]]],
            [[[-3.839926e-05, -8.607108e+02, 1.515257e-05],
              [-8.607108e+02, -2.006040e-05, -1.040868e-05],
              [1.515257e-05, -1.040868e-05, -1.487595e-05]],
             [[-8.607108e+02, -2.006040e-05, -1.040868e-05],
              [-2.006040e-05, -2.229954e+03, 1.370702e-05],
              [-1.040868e-05, 1.370702e-05, -8.607109e+02]],
             [[1.515257e-05, -1.040868e-05, -1.487595e-05],
              [-1.040868e-05, 1.370702e-05, -8.607109e+02],
              [-1.487595e-05, -8.607109e+02, 4.011427e-05]]],
            [[[-3.112595e-05, 1.515257e-05, -8.607108e+02],
              [1.515257e-05, -1.040868e-05, -1.487595e-05],
              [-8.607108e+02, -1.487595e-05, -5.166972e-05]],
             [[1.515257e-05, -1.040868e-05, -1.487595e-05],
              [-1.040868e-05, 1.370702e-05, -8.607109e+02],
              [-1.487595e-05, -8.607109e+02, 4.011427e-05]],
             [[-8.607108e+02, -1.487595e-05, -5.166972e-05],
              [-1.487595e-05, -8.607109e+02, 4.011427e-05],
              [-5.166972e-05, 4.011427e-05, -2.229954e+03]]]
        ])

        orig_g = derivatives_e.SecondHyperpolarizabilityTensor(tensor=gamma)

        for angles in angles_set:
            new_gamma = tensor_rotate(gamma, *angles)
            ng = derivatives_e.SecondHyperpolarizabilityTensor(tensor=new_gamma)

            self.assertAlmostEqual(ng.gamma_parallel(), orig_g.gamma_parallel(), places=3)
            self.assertAlmostEqual(ng.gamma_perpendicular(), orig_g.gamma_perpendicular(), places=3)
            self.assertAlmostEqual(ng.gamma_kerr(), orig_g.gamma_kerr(), places=3)

            self.assertAlmostEqual(ng.gamma_squared_zzzz(), orig_g.gamma_squared_zzzz(), delta=10)
            self.assertAlmostEqual(ng.gamma_squared_zxxx(), orig_g.gamma_squared_zxxx(), delta=10)
            self.assertAlmostEqual(ng.gamma_ths(), orig_g.gamma_ths(), places=3)
            self.assertAlmostEqual(ng.depolarization_ratio(), orig_g.depolarization_ratio(), places=3)

            self.assertAlmostEqual(
                ng.isotropic_contribution_squared(old_version=True),
                ng.isotropic_contribution_squared(old_version=False),
                places=4
            )

            self.assertAlmostEqual(
                ng.hexadecapolar_contribution_squared(old_version=True),
                ng.hexadecapolar_contribution_squared(old_version=False),
                places=4
            )

            self.assertAlmostEqual(ng.spherical_J2_contribution_squared(), .0, places=3)
            self.assertAlmostEqual(ng.spherical_J1_contribution_squared(), .0, places=3)
            self.assertAlmostEqual(ng.spherical_J3_contribution_squared(), .0, places=3)

        # water, 514.5nm, CCSD/aug-cc-pVDZ (dalton)
        gamma = numpy.array([
            [[[-2.159784e+04, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, -1.757685e+04, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -1.229660e+04]],
             [[0.000000e+00, -1.757685e+04, 0.000000e+00],
              [-1.757685e+04, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, 0.000000e+00]],
             [[0.000000e+00, 0.000000e+00, -1.229660e+04],
              [0.000000e+00, 0.000000e+00, 0.000000e+00],
              [-1.229660e+04, 0.000000e+00, 0.000000e+00]]],
            [[[0.000000e+00, -7.928857e+02, 0.000000e+00],
              [-7.928857e+02, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, 0.000000e+00]],
             [[-7.928857e+02, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, -8.872187e+02, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -6.652815e+02]],
             [[0.000000e+00, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -6.652815e+02],
              [0.000000e+00, -6.652815e+02, 0.000000e+00]]],
            [[[0.000000e+00, 0.000000e+00, -8.213404e+02],
              [0.000000e+00, 0.000000e+00, 0.000000e+00],
              [-8.213404e+02, 0.000000e+00, 0.000000e+00]],
             [[0.000000e+00, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -1.166566e+03],
              [0.000000e+00, -1.166566e+03, 0.000000e+00]],
             [[-8.213404e+02, 0.000000e+00, 0.000000e+00],
              [0.000000e+00, -1.166566e+03, 0.000000e+00],
              [0.000000e+00, 0.000000e+00, -2.396609e+03]]]
        ])

        orig_g = derivatives_e.SecondHyperpolarizabilityTensor(tensor=gamma)

        for angles in angles_set:
            new_gamma = tensor_rotate(gamma, *angles)
            ng = derivatives_e.SecondHyperpolarizabilityTensor(tensor=new_gamma)

            self.assertAlmostEqual(ng.gamma_parallel(), orig_g.gamma_parallel(), places=2)
            self.assertAlmostEqual(ng.gamma_perpendicular(), orig_g.gamma_perpendicular(), places=2)
            self.assertAlmostEqual(ng.gamma_kerr(), orig_g.gamma_kerr(), places=2)

            self.assertAlmostEqual(ng.gamma_squared_zzzz(), orig_g.gamma_squared_zzzz(), delta=100)
            self.assertAlmostEqual(ng.gamma_squared_zxxx(), orig_g.gamma_squared_zxxx(), delta=100)
            self.assertAlmostEqual(ng.gamma_ths(), orig_g.gamma_ths(), places=1)
            self.assertAlmostEqual(ng.depolarization_ratio(), orig_g.depolarization_ratio(), places=1)

            self.assertAlmostEqual(
                ng.isotropic_contribution_squared(old_version=True),
                ng.isotropic_contribution_squared(old_version=False),
                places=3
            )  # Kleinman's conditions does not change anything here!

            self.assertNotEqual(
                ng.quadrupolar_contribution_squared(old_version=True),
                ng.quadrupolar_contribution_squared(old_version=False)
            )

            self.assertNotEqual(
                ng.hexadecapolar_contribution_squared(old_version=True),
                ng.hexadecapolar_contribution_squared(old_version=False)
            )

            self.assertAlmostEqual(ng.spherical_J1_contribution_squared(), .0, places=3)
            self.assertNotEqual(ng.spherical_J3_contribution_squared(), .0)

        # test conversion
        self.assertAlmostEqual(derivatives_e.convert_frequency_from_string('1064nm'), 0.0428, places=3)
        self.assertAlmostEqual(derivatives_e.convert_frequency_from_string('2eV'), 0.073, places=3)
        self.assertAlmostEqual(derivatives_e.convert_frequency_from_string('1500cm-1'), 0.0068, places=4)

        # order and name
        g = derivatives_e.BaseElectricalDerivativeTensor(input_fields=(1, 0))
        self.assertEqual(g.representation.representation(), 'dDF')
        self.assertEqual(g.name, 'beta(-w;w,0)')
        self.assertEqual(g.rank(), 3)

        # just check that DFWM is now possible:
        g = derivatives_e.BaseElectricalDerivativeTensor(input_fields=(1, 1, -1))
        self.assertEqual(g.representation.representation(), 'dDDd')
        self.assertEqual(g.name, 'gamma(-w;w,w,-w)')
        self.assertEqual(g.rank(), 4)

        g = derivatives_e.BaseElectricalDerivativeTensor(input_fields=(-1, 1, 1))
        self.assertEqual(g.representation.representation(), 'dDDd')  # reordering
        self.assertEqual(g.name, 'gamma(-w;w,w,-w)')

    def test_electric_responses(self):
        """Check responses to electric field"""

        b_tensor = numpy.zeros((3, 3, 3))
        b_tensor[0, 0, 0] = 1.
        b_tensor[1, 1, 1] = 2.
        b_tensor[2, 2, 2] = 3.

        b = derivatives_e.FirstHyperpolarisabilityTensor(tensor=b_tensor)
        self.assertArraysAlmostEqual(b.response_to_electric_field([1, 0, 0]), [1., 0, 0])
        self.assertArraysAlmostEqual(b.response_to_electric_field([0, 1, 0]), [0, 2., 0])
        self.assertArraysAlmostEqual(b.response_to_electric_field([0, 0, 1]), [0, 0, 3.])

    def test_geometrical_derivatives(self):
        """Test geometrical ones.

        Note: only tests the Hessian
        """

        # H2O molecule:
        atom_list = [
            qcip_atom.Atom(symbol='O', position=[0, .0, 0.11393182]),
            qcip_atom.Atom(symbol='H', position=[0, -0.75394266, -0.45572727]),
            qcip_atom.Atom(symbol='H', position=[0, 0.75394266, -0.45572727])
        ]

        water_molecule = qcip_molecule.Molecule(atom_list=atom_list)

        # Water (geometry above) HF/Sadlej-POL (cartesian set)
        hessian = numpy.array(
            [[-0.00011, .0, .0, 0.00005, .0, .0, 0.00005, .0, .0],
             [.0, 0.80982, .0, .0, -0.40491, -0.30598, .0, -0.40491, 0.30598],
             [.0, .0, 0.53532, .0, -0.23905, -0.26766, .0, 0.23905, -0.26766],
             [0.00005, .0, .0, .0, .0, .0, -0.00005, .0, .0],
             [.0, -0.40491, -0.23905, .0, 0.43721, 0.27252, .0, -0.03230, -0.03346],
             [.0, -0.30598, -0.26766, .0, 0.27252, 0.24945, .0, 0.03346, 0.01821],
             [0.00005, .0, .0, -0.00005, .0, .0, .0, .0, .0],
             [.0, -0.40491, 0.23905, .0, -0.03230, 0.03346, .0, 0.43721, -0.27252],
             [.0, 0.30598, -0.26766, .0, -0.03346, 0.01821, .0, -0.27252, 0.24945]]
        )

        h = derivatives_g.BaseGeometricalDerivativeTensor(
            3 * len(water_molecule), 5 if water_molecule.linear() else 6, 'GG', components=hessian)

        mwh = derivatives_g.MassWeightedHessian(water_molecule, hessian)
        self.assertEqual(mwh.vibrational_dof, 3)
        self.assertFalse(mwh.linear)

        self.assertEqual(len(mwh.frequencies), 9)

        self.assertArraysAlmostEqual(
            [-38.44, -3.8, -0.03, 0.0, 18.1, 36.2, 1760.4, 4136.5, 4244.3],
            [a * 219474.63 for a in mwh.frequencies],
            places=1)

        # projected hessian must contain square of frequency on the diagonal
        projected_h = h.project_over_normal_modes(mwh)
        self.assertEqual(projected_h.representation.representation(), 'NN')  # change type

        for i in range(h.spacial_dof):
            self.assertAlmostEqual(
                math.fabs(projected_h.components[i, i]), mwh.frequencies[i] ** 2, places=10)

        # check output
        self.assertIn('dQ(1)', projected_h.to_string())
        self.assertNotIn('dQ(1)', projected_h.to_string(skip_trans_plus_rot_dof=6))


class TensorNumDiff(QcipToolsTestCase):

    def test_numerical_differentiation_F(self):
        energy = 150.
        mu = factories.FakeElectricDipole()
        alpha = factories.FakePolarizabilityTensor(input_fields=(0,))
        beta = factories.FakeFirstHyperpolarizabilityTensor(input_fields=(0, 0))

        min_field = 0.004
        k_max = 5
        ratio = 2.

        def energy_exp(fields, h0, basis, component, **kwargs):
            """Taylor series of the energy"""

            r_field = numerical_differentiation.real_fields(fields, h0, ratio)

            x = energy
            x -= numpy.tensordot(mu.components, r_field, axes=1)
            x -= 1 / 2 * numpy.tensordot(numpy.tensordot(alpha.components, r_field, axes=1), r_field, axes=1)
            x -= 1 / 6 * numpy.tensordot(
                numpy.tensordot(numpy.tensordot(beta.components, r_field, axes=1), r_field, axes=1), r_field, axes=1)

            return x

        def dipole_exp(fields, h0, basis, component, **kwargs):
            """Taylor series of the dipole moment"""

            r_field = numerical_differentiation.real_fields(fields, h0, ratio)

            x = mu.components.copy()
            x += numpy.tensordot(alpha.components, r_field, axes=1)
            x += 1 / 2 * numpy.tensordot(numpy.tensordot(beta.components, r_field, axes=1), r_field, axes=1)

            return x[component]

        # compute polarizability
        t, triangles = derivatives.compute_numerical_derivative_of_tensor(
            derivatives.Derivative(from_representation='F'),
            derivatives.Derivative('F'),
            dipole_exp, k_max, min_field, ratio)

        self.assertArraysAlmostEqual(alpha.components, t.components, places=3)

        t, triangles = derivatives.compute_numerical_derivative_of_tensor(
            derivatives.Derivative(from_representation=''),
            derivatives.Derivative('FF'),
            energy_exp, k_max, min_field, ratio)

        self.assertArraysAlmostEqual(alpha.components, t.components, places=3)

        # compute first polarizability
        t, triangles = derivatives.compute_numerical_derivative_of_tensor(
            derivatives.Derivative(from_representation='F'),
            derivatives.Derivative('FF'),
            dipole_exp, k_max, min_field, ratio)

        self.assertArraysAlmostEqual(beta.components, t.components, delta=.001)

        t, triangles = derivatives.compute_numerical_derivative_of_tensor(
            derivatives.Derivative(from_representation=''),
            derivatives.Derivative('FFF'),
            energy_exp, k_max, min_field, ratio)

        self.assertArraysAlmostEqual(beta.components, t.components, delta=.01)

    def test_numerical_differentiation_G(self):
        input_fields = (0, 1)
        d = ''.join('D' if a == 1 else 'F' for a in input_fields)

        s = -sum(input_fields)
        if s in derivatives_e.field_to_representation:
            d = derivatives_e.field_to_representation[s] + d
        else:
            d = 'X' + d

        t_fdf = factories.FakeFirstHyperpolarizabilityTensor(input_fields=input_fields, frequency=.1)
        t_nfdf = factories.FakeTensor(derivatives.Derivative('N' + d, spacial_dof=6), frequency=.1, factor=10.)
        t_nnfdf = factories.FakeTensor(derivatives.Derivative('NN' + d, spacial_dof=6), frequency=.1, factor=100.)
        t_nnnfdf = factories.FakeTensor(derivatives.Derivative('NNN' + d, spacial_dof=6), frequency=.1, factor=1000.)

        min_field = 0.01
        k_max = 5
        ratio = 2.

        def hp_exp(fields, h0, basis, component, **kwargs):
            r_field = numerical_differentiation.real_fields(fields, h0, ratio)

            x = t_fdf.components.copy()
            x += numpy.tensordot(r_field, t_nfdf.components, axes=1)
            x += 1 / 2 * numpy.tensordot(r_field, numpy.tensordot(r_field, t_nnfdf.components, axes=1), axes=1)
            x += 1 / 6 * numpy.tensordot(r_field, numpy.tensordot(
                r_field, numpy.tensordot(r_field, t_nnnfdf.components, axes=1), axes=1), axes=1)

            return x[component]

        t, triangles = derivatives.compute_numerical_derivative_of_tensor(
            derivatives.Derivative(from_representation=d),
            derivatives.Derivative('G', spacial_dof=6),
            hp_exp, k_max, min_field, ratio, frequency=.1)

        self.assertArraysAlmostEqual(t_nfdf.components, t.components, places=3)

        t, triangles = derivatives.compute_numerical_derivative_of_tensor(
            derivatives.Derivative(from_representation=d),
            derivatives.Derivative('GG', spacial_dof=6),
            hp_exp, k_max, min_field, ratio, frequency=.1)

        self.assertArraysAlmostEqual(t_nnfdf.components, t.components, places=3)
