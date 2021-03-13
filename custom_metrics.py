import sympy as sp
import numpy as np
from IPython.display import display, Latex

# Asymmetric loss ASL
class AsymmetricLossObjective(object):
    def __init__(self, der1, der2, gamma_minus=.2, gamma_plus=.6):
        self.g_minus = gamma_minus
        self.g_plus = gamma_plus
        self.der1 = der1
        self.der2 = der2

    def calc_ders_range(self, x, y, weights):
        assert len(x) == len(y)
        if weights is not None:
            assert len(weights) == len(x)

        e = np.exp(x)
        p = e / (1 + e)
        log_p = np.log(p)
        log_1minus_p = np.log(1-p)

        der1 = self.der1(p=p, y=y, g_minus=self.g_minus, g_plus=self.g_plus, log_p=log_p, log_1minus_p=log_1minus_p)
        der2 = self.der2(p=p, y=y, g_minus=self.g_minus, g_plus=self.g_plus, log_p=log_p, log_1minus_p=log_1minus_p)

        # der1 = self.der1(p=p, y=y, g_minus=self.g_minus, g_plus=self.g_plus)
        # der2 = self.der2(p=p, y=y, g_minus=self.g_minus, g_plus=self.g_plus)

        if weights is not None:
            der1 *= weights
            der2 *= weights

        result = list(zip(der1, der2))

        return result


class LoglossObjective_loop(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = targets[index] - p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


#%%

class LoglossObjective_np(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)


        e = np.exp(approxes)
        p = e / (1 + e)
        der1 = targets - p
        der2 = -p * (1 - p)

        if weights is not None:
            der1 *= weights
            der2 *= weights


        result = list(zip(der1, der2))
        return result

def FocalLossFormulas():
    x, y, gamma = sp.symbols('x y gamma')

    p = sp.exp(x) / (1 + sp.exp(x))
    display(sp.Eq(sp.S('p'), p))

    L_plus = (1 - p)**gamma * sp.log(p)
    L_minus = p**gamma * sp.log(1 - p)

    display(sp.Eq(sp.S('L_plus'), L_plus))
    display(sp.Eq(sp.S('L_minus'), L_minus))

    L = -y*L_plus - (1 - y)*L_minus
    display(sp.Eq(sp.S('L'), L))

    der1 = sp.diff(L, x)
    display(sp.Eq(sp.S('L__der1'), der1))

    der2 = sp.diff(der1, x)
    display(sp.Eq(sp.S('L__der2'), der2))
    return der1, der2

# der1, der2 = FocalLossFormulas()

def AsymmetricLossFormulas():
    x, y, g_minus, g_plus = sp.symbols('x y g_minus g_plus')

    p = sp.exp(x) / (1 + sp.exp(x))
    display(sp.Eq(sp.S('p'), p))

    L_plus = (1 - p)**g_plus * sp.log(p)
    L_minus = p**g_minus * sp.log(1 - p)

    display(sp.Eq(sp.S('L_plus'), L_plus))
    display(sp.Eq(sp.S('L_minus'), L_minus))

    L = -y*L_plus - (1 - y)*L_minus
    display(sp.Eq(sp.S('L'), L))

    der1 = sp.diff(L, x)
    display(sp.Eq(sp.S('L__der1'), der1))

    der2 = sp.diff(der1, x)
    display(sp.Eq(sp.S('L__der2'), der2))
    return der1, der2



def get_simplified_derivative(derivative, verbose=False, der_name='ALS_der'):
    """
    Упрощает выражение для производной методами библиотеки sympy

    :param derivative: функция производной, полученная методом sympy.diff()
    :param verbose: Вывести промежуточные формулы на экран, могут быть длинными
    :return: Функция производной, оптимизированная для numpy
    """

    def print_der(der, message):
        if verbose:
            display(message)
            display(sp.Eq(sp.S(der_name), der))

    p, x, y = sp.symbols('p x y')
    g_minus, g_plus = sp.symbols('g_minus g_plus')
    log_p, log_1minus_p = sp.symbols('log_p log_1minus_p')

    # log_p, log_1minus_p = sp.symbols(log_p, log_1minus_p)
    der = derivative

    print_der(derivative, 'Входящая функция производной')

    der = der.subs(sp.exp(x) / (1 + sp.exp(x)), p)
    print_der(der, '*********** der.subs(sp.exp(x) / (1 + sp.exp(x)), p) *************')

    der = der.subs(sp.exp(x) / (sp.exp(x) + 1), p)
    print_der(der, '*********** der.subs(sp.exp(x) / (1 + sp.exp(x)), p) *************')

    der = der.subs((1 + sp.exp(x)) ** 2 * sp.exp(x) ** (-2), p ** (-2))
    print_der(der, '**********    der.subs((1 + sp.exp(x))**2 * sp.exp(x)**(-2), p**(-2))  *****************')

    der = der.subs(sp.exp(x), p / (1 - p))
    print_der(der, '**********    der.subs(sp.exp(x), p / (1-p))  *****************')

    der = der.subs(sp.log(p), log_p)
    print_der(der, '**********    der.subs(sp.log(p), log_p)  *****************')

    der = der.subs(sp.log(1-p), log_1minus_p)
    print_der(der, '**********    der.subs(sp.log(1-p), log_1minus_p)  *****************')

    der = sp.simplify(der)
    print_der(der, '*********** sp.symplify(der) *************')

    der = sp.factor(der)
    print_der(der, '*********** sp.factor(der) *************')

    der_lambdify = sp.lambdify([p, y, g_minus, g_plus, log_p, log_1minus_p], der, 'numpy')

    return der_lambdify


# der_l2 = get_simplified_derivative(ALS_der2, verbose=True, der_name='ALS_der2')

