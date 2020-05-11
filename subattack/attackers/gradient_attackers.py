import torch

from subattack.strategies.adv_checkers import AdvChecker
from subattack.strategies.initializers import Initializer
from subattack.strategies.loss import LossEvaluator
from subattack.strategies.conventions import Convention
from subattack.strategies.constraints import Constraint
from subattack.strategies.gradient_estimators import GradientEstimator
from subattack.strategies.steepest import SteepestGradientTransformer


class GradientAttacker:

    def __init__(
            self,
            model,
            step_size,
            budget,
            adv_checker: AdvChecker,
            convention: Convention,
            initializer: Initializer,
            gradient_estimator: GradientEstimator,
            constraint: Constraint,
            loss_evaluator: LossEvaluator,
            steepest_gradient_transformer: SteepestGradientTransformer,
            verbose=True,
    ):
        self._model = model
        self._step_size = step_size
        self._budget = budget
        self._adv_checker = adv_checker
        self._convention = convention
        self._initializer = initializer
        self._gradient_estimator = gradient_estimator
        self._constraint = constraint
        self._loss_evaluator = loss_evaluator
        self._steepest_gradient_transformer = steepest_gradient_transformer
        self._verbose = verbose

    def solve(self, image, label):

        perturbation = self._initialize(image)
        adv_image = perturbation + image

        total_cost = 0
        while True:

            pred_label = self._predict_label(adv_image)
            total_cost += 1

            if self._verbose:
                loss = self._loss_evaluator.compute_individual(
                    self._model, adv_image, label
                )
                print('{:>5d} {:>10.6f}'.format(total_cost, loss.item()))

            if ((self._adv_checker.successful(label, pred_label)) or
                    (total_cost >= self._budget)):

                return adv_image, pred_label, total_cost

            normalized_gradient, gradient_cost = self._estimate_gradient(
                adv_image, label
            )

            total_cost += gradient_cost
            increment = normalized_gradient * self._step_size

            perturbation = self._update(image, perturbation, increment)
            adv_image = image + perturbation

    def _initialize(self, image):
        return self._project(
            self._initializer.initialize(), image
        )

    def _project(self, perturbation, image):
        perturbation = self._project_perturbation(perturbation)
        adv_image = self._project_image(image + perturbation)
        perturbation = adv_image - image
        return perturbation

    def _project_image(self, image):
        return self._convention.project(image)

    def _project_perturbation(self, perturbation):
        return self._constraint.project(perturbation)

    def _predict_label(self, image):
        return self._model(image.unsqueeze(0)).squeeze().argmax()

    def _estimate_gradient(self, image, label):
        gradient, gradient_cost = self._gradient_estimator.estimate(
            self._model, self._loss_evaluator, image, label
        )
        steepest_gradient = self._steepest_gradient_transformer.transform(
            gradient
        )
        return steepest_gradient, gradient_cost

    def _update(self, image, perturbation, increment):
        cand_perturbation = self._project(perturbation + increment, image)

        if torch.allclose(perturbation, cand_perturbation):
            perturbation = self._initialize(image)
        else:
            perturbation = cand_perturbation
        return perturbation
