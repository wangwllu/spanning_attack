import torch
from abc import ABC, abstractmethod

from subattack.utils import Predictor
from subattack.strategies import Initializer
from subattack.strategies import Sampler
from subattack.strategies import Convention
from subattack.strategies import AdvChecker


class AbstractBoundaryAttacker(ABC):

    def __init__(
            self, predictor: Predictor,
            outer_ratio, inner_ratio, budget,
            convention: Convention,
            initializer: Initializer,
            sampler: Sampler,
            adv_checker: AdvChecker,
            verbose=True,
    ):
        self._predictor = predictor

        self._outer_ratio = outer_ratio
        self._inner_ratio = inner_ratio
        self._budget = budget

        self._convention = convention
        self._initializer = initializer
        self._sampler = sampler
        self._adv_checker = adv_checker

        self._verbose = verbose

        self._cost = 0

    def solve(self, image, label):

        self._cost = 0

        try:
            adv_image = self._initialize(image, label)
        except InitializationException:
            return image

        if self._verbose:
            print(
                f'{self.cost:5d}, {(adv_image-image).norm().item():f}'
            )

        while True:

            if self._ready_for_early_stop(image, label, adv_image):
                return adv_image

            if self.cost >= self._budget:
                if self._constraint_satisfied(image, label, adv_image):
                    return adv_image
                else:
                    return image

            outer_image = self._outer_move(image, adv_image)
            inner_image = self._inner_move(image, outer_image)

            if self._adv_checker.successful(
                    label, self._predict(inner_image)
            ):
                adv_image = inner_image

                if self._verbose:
                    print(
                        f'{self.cost:5d}, {(adv_image-image).norm().item():f}'
                    )

    def _initialize(self, image, label):
        while True:

            if self.cost >= self._budget:
                raise InitializationException('budget ran out')

            result = self._convention.project(
                image + self._initializer.initialize()
            )
            if self._adv_checker.successful(
                    self._predict(result), label
            ):
                return result

    @abstractmethod
    def _ready_for_early_stop(self, image, label, adv_image):
        pass

    @abstractmethod
    def _constraint_satisfied(self, image, label, adv_image):
        pass

    def _predict(self, image):
        self._cost += 1
        return self._predict_for_free(image)

    def _predict_for_free(self, image):
        return self._predictor.predict_individual(image)

    def _outer_move(self, image, adv_image):

        old_perturbation_norm = (adv_image - image).norm()
        random_move = (
            self._outer_ratio
            * old_perturbation_norm
            * self._sampler.sample(1).squeeze(0)
        )
        perturbation = adv_image + random_move - image
        result = image + (
            old_perturbation_norm * perturbation
            / perturbation.norm().clamp(min=1e-7)
        )
        return self._convention.project(result)

    def _inner_move(self, image, outer_image):
        return outer_image + self._inner_ratio * (image - outer_image)

    @property
    def cost(self):
        return self._cost


class InitializationException(Exception):
    pass


class BoundaryAttacker(AbstractBoundaryAttacker):

    def _ready_for_early_stop(self, image, label, adv_image):
        return False

    def _constraint_satisfied(self, image, label, adv_image):
        return True


class BoundaryAttackerWithStopRadius(AbstractBoundaryAttacker):

    def __init__(
            self, predictor: Predictor,
            outer_ratio, inner_ratio, budget,
            radius,
            convention: Convention,
            initializer: Initializer,
            sampler: Sampler,
            adv_checker: AdvChecker,
            verbose=True,
    ):
        super().__init__(predictor, outer_ratio, inner_ratio, budget,
                         convention, initializer, sampler, adv_checker,
                         verbose)
        self._radius = radius

    def _ready_for_early_stop(self, image, label, adv_image):
        return (torch.norm(adv_image - image) <= self._radius)

    def _constraint_satisfied(self, image, label, adv_image):
        return self._ready_for_early_stop(image, label, adv_image)
