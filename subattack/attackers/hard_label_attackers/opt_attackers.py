import math

from subattack.utils.predictors import ModulePredictor
from subattack.strategies.adv_checkers import AdvChecker


class OptAttacker:

    def __init__(
            self, scale_ratio,
            predictor: ModulePredictor,
            adv_checker: AdvChecker,
    ):
        self._scale_ratio = scale_ratio
        self._inf_bound = 20

        self._predictor = predictor
        self._adv_checker = adv_checker

        self._cost = 0

    def solve(self, image, label):
        self._cost = 0

    @property
    def cost(self):
        return self._cost

    def _estimate_gradient(self, image):
        pass

    def _estimate_directional_distance(self, image, label, direction, prior):

        def successful(distance):
            return self._get_attack_feedback(
                image + distance * direction, label
            )

        lower, upper = self._scale_search(
            successful, prior, self._scale_ratio, self._inf_bound
        )
        return self._binary_search(successful, lower, upper,)

    def _get_attack_feedback(self, image, label):
        self._cost += 1
        return self._get_attack_feedback_for_free(image, label)

    def _get_attack_feedback_for_free(self, image, label):
        return self._adv_checker.successful(
            label, self._predictor.predict_individual(image)
        )

    @staticmethod
    def _scale_search(successful, prior, scale_ratio, inf_bound=math.inf):

        # successful(0.) must be False

        upper = prior
        lower = prior

        if successful(prior):
            while successful(lower):
                lower *= (1 - scale_ratio)
        else:
            while not successful(upper):
                if upper > inf_bound:
                    return lower, math.inf
                upper *= (1 + scale_ratio)

        return lower, upper

    @staticmethod
    def _binary_search(successful, lower, upper, abs_tol=1e-9):

        # successful(lower) must be False
        # successful(upper) must be True

        if math.isinf(upper):
            return upper

        while not math.isclose(lower, upper, abs_tol=abs_tol):
            mid = (lower + upper) / 2
            if successful(mid):
                upper = mid
            else:
                lower = mid

        return upper
