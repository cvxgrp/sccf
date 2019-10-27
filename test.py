import unittest

import sccf
import cvxpy as cp
import numpy as np


class TestMinExpression(unittest.TestCase):
    def test(self):
        x = cp.Variable(10)
        x.value = np.zeros(10)
        expr = cp.sum_squares(x)
        with self.assertRaises(AssertionError):
            sccf.minimum(-expr, 1.0)
        min_expr = sccf.minimum(expr, 1.0)
        self.assertEqual(min_expr.value, 0.0)
        x.value = np.ones(10)
        self.assertEqual(min_expr.value, 1.0)


class TestSumOfMinExpressions(unittest.TestCase):
    def test(self):
        x = cp.Variable(10)
        x.value = np.zeros(10)
        expr1 = sccf.minimum(cp.sum_squares(x), 1.0)
        expr2 = sccf.minimum(cp.sum_squares(x - 1), 1.0)
        sum_exprs = expr1 + expr2
        self.assertEqual(len(sum_exprs.min_exprs), 2)
        self.assertEqual(sum_exprs.value, 1.0)
        sum_exprs += 1.0
        self.assertEqual(sum_exprs.value, 2.0)
        sum_exprs += cp.sum_squares(x - 1)
        self.assertEqual(sum_exprs.value, 12.0)


class TestProblem(unittest.TestCase):
    def test(self):
        x = cp.Variable(1)
        x.value = 2*np.ones(1)
        expr1 = sccf.minimum(cp.sum(cp.huber(x)), 1.0)
        expr2 = sccf.minimum(cp.sum(cp.huber(x - 1.0)), 1.0)
        obj = expr1 + expr2
        starting = obj.value
        prob = sccf.Problem(obj)
        ending = prob.solve()
        self.assertLessEqual(ending["final_objective_value"], starting)

if __name__ == '__main__':
    unittest.main()