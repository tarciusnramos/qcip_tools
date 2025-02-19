import numpy
import math

from tests import QcipToolsTestCase

from qcip_tools import numerical_differentiation


class NumericalDifferentiationTestCase(QcipToolsTestCase):

    def setUp(self):
        pass

    coefficients_of_univariate_polynom = [3, 15, -250, 1240, -10450]  # one order of magnitude between each coef.

    @staticmethod
    def univariate_polynom(x, coefficients=coefficients_of_univariate_polynom):
        accum = .0
        for i in range(len(coefficients)):
            accum += 1 / math.factorial(i) * coefficients[i] * x ** i

        return accum

    def test_coefficients(self):
        """Ensure the right coeficients are found"""

        self.assertEqual(self.univariate_polynom(0), self.coefficients_of_univariate_polynom[0])

        # Test some well-known derivatives formulas:
        c = numerical_differentiation.Coefficients(1, 1, method='F')
        self.assertTrue(numpy.array_equal(c.mat_coefs, [-1, 1]))
        self.assertTrue(numpy.array_equal(c.mat_i, [0, 1]))

        c = numerical_differentiation.Coefficients(1, 1, method='B')
        self.assertTrue(numpy.array_equal(c.mat_coefs, [-1, 1]))
        self.assertTrue(numpy.array_equal(c.mat_i, [-1, 0]))

        c = numerical_differentiation.Coefficients(1, 2, method='C')
        self.assertTrue(numpy.array_equal(c.mat_coefs, [-.5, 0, .5]))
        self.assertTrue(numpy.array_equal(c.mat_i, [-1, 0, 1]))

        # Test for different values of the ratio, different minimal values and different derivatives
        for a in [0.5, 2.0, 2.0**(1 / 2), 2**(1 / 3), 3, 10]:

            self.assertEqual(numerical_differentiation.ak_shifted(a, 0), 0)
            self.assertEqual(numerical_differentiation.ak_shifted(a, 1), a**0)
            self.assertEqual(numerical_differentiation.ak_shifted(a, -1), -a**0)
            self.assertEqual(numerical_differentiation.ak_shifted(a, 2), a**1)
            self.assertEqual(numerical_differentiation.ak_shifted(a, -2), -a**1)
            self.assertEqual(numerical_differentiation.ak_shifted(a, 3), a**2)
            self.assertEqual(numerical_differentiation.ak_shifted(a, -3), -a**2)

            for h0 in [0.01, 0.001]:
                for d in [1, 2, 3]:

                    # test the different methods:
                    for method in ['F', 'B', 'C']:

                        c = numerical_differentiation.Coefficients(
                            d,
                            numerical_differentiation.Coefficients.choose_p_for_centered(d) if method == 'C' else 3,
                            ratio=a,
                            method=method)

                        derivative = \
                            c.prefactor(0, h0) * sum(
                                c.mat_coefs[i] * self.univariate_polynom(
                                    numerical_differentiation.ak_shifted(a, c.mat_i[i]) * h0)
                                for i in range(len(c.mat_coefs)))

                        # notice the large threshold:
                        self.assertAlmostEqual(
                            derivative, self.coefficients_of_univariate_polynom[d], delta=1)

    def test_numerical_differentiation_of_univariate_function(self):
        """Same kind of test that in test_coefficients(), but directly with compute_derivative_of_function()"""

        other_coefficients = [-6, -13, 275, 2430, -9825]

        def scalar_function(fields, h_0, a, coefficients=self.coefficients_of_univariate_polynom):
            return self.univariate_polynom(numerical_differentiation.ak_shifted(a, fields[0]) * h_0, coefficients)

        # Test for different values of the ratio, different minimal values and amplitudes and different derivatives
        for a in [0.5, 2.0, 2.0 ** (1 / 2), 2 ** (1 / 3), 3]:
            for h0 in [0.01, 0.001]:
                for d in [1, 2, 3]:
                    for method in ['F', 'B', 'C']:
                        c = numerical_differentiation.Coefficients(
                            d,
                            numerical_differentiation.Coefficients.choose_p_for_centered(d) if method == 'C' else 3,
                            ratio=a,
                            method=method)

                        for k in [0, 1]:

                            # with default coefficients:
                            derivative = numerical_differentiation.compute_derivative_of_function(
                                [(c, 0)],
                                scalar_function,
                                k,
                                h0,
                                1,  # input space = 1 because function of R → R
                                a=a
                            )

                            self.assertAlmostEqual(
                                derivative,
                                self.coefficients_of_univariate_polynom[d],
                                delta=5)

                            # with custom coefficients
                            derivative = numerical_differentiation.compute_derivative_of_function(
                                [(c, 0)],
                                scalar_function,
                                k,
                                h0,
                                1,
                                a=a,
                                coefficients=other_coefficients  # test argument passing
                            )

                            self.assertNotEqual(other_coefficients[d], self.coefficients_of_univariate_polynom[d])

                            self.assertAlmostEqual(
                                derivative,
                                other_coefficients[d],
                                delta=5)

    coefficients_of_bivariate_polynom = [
        5,
        numpy.array([12, -6]),
        numpy.array([[120, -14], [-14, -175]]),  # symmetric tensor
        numpy.array([[[1420, 6400], [6400, -475]], [[6400, -475], [-475, -4320]]])
    ]

    @staticmethod
    def bivariate_polynom(x, coefficients=coefficients_of_bivariate_polynom):
        accum = .0
        for i in range(len(coefficients)):
            if i == 0:
                accum = coefficients[0]
                continue

            accum2 = 1 / math.factorial(i) * numpy.tensordot(coefficients[i], x, axes=1)
            for j in range(i - 1):
                accum2 = numpy.tensordot(accum2, x, axes=1)

            accum += accum2

        return accum

    def test_numerical_differentiation_of_bivariate_function(self):
        """Test for bivariate function (a bit simpler than the one for univariate function)"""

        def scalar_function(fields, h_0, a, coefficients=self.coefficients_of_bivariate_polynom):
            return self.bivariate_polynom(
                [numerical_differentiation.ak_shifted(a, i_) * h_0 for i_ in fields], coefficients)

        coefficients = [
            numerical_differentiation.Coefficients(1, 2, ratio=2, method='C'),
            numerical_differentiation.Coefficients(2, 1, ratio=2, method='C'),
            numerical_differentiation.Coefficients(3, 2, ratio=2, method='C')
        ]

        # first order:
        for i in range(2):

            derivative = numerical_differentiation.compute_derivative_of_function(
                [(coefficients[0], i)],
                scalar_function,
                0,
                0.01,
                2,  # function R² → R, so input space size = 2.
                a=2
            )

            self.assertAlmostEqual(derivative, self.coefficients_of_bivariate_polynom[1][i], delta=1)

        # second order:
        for i in range(2):
            for j in range(i + 1):

                if i != j:
                    derivative = numerical_differentiation.compute_derivative_of_function(
                        [(coefficients[0], i), (coefficients[0], j)],
                        scalar_function,
                        0,
                        0.01,
                        2,
                        a=2
                    )

                else:
                    derivative = numerical_differentiation.compute_derivative_of_function(
                        [(coefficients[1], i)],
                        scalar_function,
                        0,
                        0.01,
                        2,
                        a=2
                    )

                self.assertAlmostEqual(derivative, self.coefficients_of_bivariate_polynom[2][i, j], delta=1)

        # third order:
        for i in range(2):
            for j in range(i + 1):
                for k in range(j + 1):

                    if i != j:  # ... then j == k
                        derivative = numerical_differentiation.compute_derivative_of_function(
                            [(coefficients[0], i), (coefficients[1], j)],
                            scalar_function,
                            0,
                            0.01,
                            2,
                            a=2
                        )

                    elif j != k:  # ... then i == j
                        derivative = numerical_differentiation.compute_derivative_of_function(
                            [(coefficients[1], i), (coefficients[0], k)],
                            scalar_function,
                            0,
                            0.01,
                            2,
                            a=2
                        )
                    else:  # ... then i == j == k
                        derivative = numerical_differentiation.compute_derivative_of_function(
                            [(coefficients[2], i)],
                            scalar_function,
                            0,
                            0.01,
                            2,
                            a=2
                        )

                    self.assertAlmostEqual(
                        derivative, self.coefficients_of_bivariate_polynom[3][i, j, k], delta=1)

    def test_romberg(self):
        """Test the Romberg's scheme implementation"""

        def scalar_function(fields, h_0, a, coefficients=self.coefficients_of_univariate_polynom):
            return self.univariate_polynom(numerical_differentiation.ak_shifted(a, fields[0]) * h_0, coefficients)

        # forward derivative gives awful results by default
        a = 2.0
        c = numerical_differentiation.Coefficients(1, 1, ratio=a, method='F')
        derivative_values = []

        for k in range(5):
            derivative_values.append(numerical_differentiation.compute_derivative_of_function(
                [(c, 0)],
                scalar_function,
                k,
                0.001,
                a=a
            ))

        t = numerical_differentiation.RombergTriangle(derivative_values, ratio=a)

        # test amplitude and iteration error
        self.assertEqual(t.amplitude_error(1, 0), t.romberg_triangle[1, 0] - t.romberg_triangle[0, 0])
        self.assertEqual(t.iteration_error(1, 0), t.romberg_triangle[0, 1] - t.romberg_triangle[0, 0])

        # test best value:
        position, value, iteration_error = t(threshold=1e-5)
        self.assertAlmostEqual(value, self.coefficients_of_univariate_polynom[1], places=5)
        self.assertTrue(position[1] > 0)  # it needed improvements !
        self.assertTrue(math.fabs(iteration_error) < 1e-3)

        # test error:
        with self.assertRaises(ValueError):
            t.amplitude_error(k=-1, m=0)

        with self.assertRaises(ValueError):
            t.amplitude_error(k=0, m=-1)

        with self.assertRaises(ValueError):
            t.amplitude_error(k=2, m=3)  # no k=2 if m=3

        with self.assertRaises(ValueError):
            t.iteration_error(k=2, m=3)  # no k=2 if m=3

    def test_romberg_force_choice(self):

        def scalar_function(fields, h_0, a, coefficients=self.coefficients_of_univariate_polynom):
            return self.univariate_polynom(numerical_differentiation.ak_shifted(a, fields[0]) * h_0, coefficients)

        # forward derivative gives awful results by default
        a = 2.0
        c = numerical_differentiation.Coefficients(1, 1, ratio=a, method='F')
        derivative_values = []

        for k in range(5):
            derivative_values.append(numerical_differentiation.compute_derivative_of_function(
                [(c, 0)],
                scalar_function,
                k,
                0.001,
                a=a
            ))

        force = (2, 2)
        t = numerical_differentiation.RombergTriangle(derivative_values, ratio=a, force_choice=force)
        position, value, iteration_error = t(threshold=1e-5)

        self.assertEqual(position, force)
