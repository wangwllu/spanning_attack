from abc import ABC, abstractmethod
import torch

from subattack.strategies.samplers import Sampler
from subattack.strategies.loss import LossEvaluator


class GradientEstimator(ABC):

    @abstractmethod
    def estimate(
        model,
        loss_evaluator: LossEvaluator,
        image, label
    ):
        pass


class RgfEstimator(GradientEstimator):

    def __init__(
        self,
        sampler: Sampler,
        change,
        sample_size,
    ):
        self._sampler = sampler
        self._change = change
        self._sample_size = sample_size

    def estimate(
        self, model,
        loss_evaluator: LossEvaluator,
        image, label,
    ):

        unit_vectors = self._sampler.sample(self._sample_size)

        gradient = (
            (
                loss_evaluator.compute_batch(
                    model,
                    image + unit_vectors * self._change,
                    label.expand(unit_vectors.shape[0])
                ) - loss_evaluator.compute_individual(model, image, label)
            ) / self._change @ unit_vectors.view(unit_vectors.shape[0], -1)
        ) / unit_vectors.shape[0]

        gradient = gradient.view(*unit_vectors.shape[1:])

        return gradient, unit_vectors.shape[0]


class PriorEstimator(GradientEstimator):

    def __init__(
            self,
            ref_model_list,
            change,
    ):
        # every model in the model list should be on cpu
        self._ref_model_list = ref_model_list
        self._change = change

    def estimate(
            self,
            model,
            loss_evaluator: LossEvaluator,
            image, label,
    ):
        ref_model = self._ref_model_list[
            torch.randint(0, len(self._ref_model_list), (1,)).item()
        ]

        ref_model.to(image.device)
        random_vector = self._compute_gradient(
            ref_model, loss_evaluator, image, label
        )
        ref_model.cpu()

        unit_vector = self._normalize_vector(random_vector)
        return unit_vector * (
            loss_evaluator.compute_individual(
                model, image + unit_vector * self._change, label
            ) - loss_evaluator.compute_individual(model, image, label)
        ) / self._change, 1

    def _compute_gradient(
            self, ref_model,
            loss_evaluator: LossEvaluator, image, label
    ):
        delta = torch.zeros_like(image, requires_grad=True)
        loss = loss_evaluator.compute_individual(ref_model, image+delta, label)
        loss.backward()
        return delta.grad.detach()

    def _normalize_vector(self, vector, epsilon=1e-7):
        return vector / vector.norm().clamp(min=epsilon)


class GradientEstimatorFactory:

    def create_gradient_estimator(
            self, name,
            sampler=None, change=None, sample_size=None,
            ref_model_list=None,
    ):
        if name == 'rgf':
            assert sampler is not None
            assert change is not None
            assert sample_size is not None
            return RgfEstimator(sampler, change, sample_size)
        elif name == 'prior':
            assert change is not None
            assert ref_model_list is not None
            return PriorEstimator(ref_model_list, change)
        else:
            raise Exception('unsupported gradient estimator')
