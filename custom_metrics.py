import numpy as np

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

