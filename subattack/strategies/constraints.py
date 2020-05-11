from abc import ABC, abstractmethod


class Constraint(ABC):

    @abstractmethod
    def is_acceptable(self, perturbation):
        pass

    @abstractmethod
    def project(self, perturbation):
        pass


class NoConstraint(Constraint):

    def is_acceptable(self, perturbation):
        return True

    def project(self, perturbation):
        return perturbation


class LinfConstraint(Constraint):

    def __init__(self, upper_bound):
        self._upper_bound = upper_bound

    def is_acceptable(self, perturbation):
        return perturbation.abs().max() <= self._upper_bound

    def project(self, perturbation):
        return perturbation.clamp(
            min=-self._upper_bound, max=self._upper_bound
        )


class L2Constraint(Constraint):

    def __init__(self, upper_bound):
        self._upper_bound = upper_bound

    def is_acceptable(self, perturbation):
        return perturbation.norm() <= self._upper_bound

    def project(self, perturbation):
        return (
            perturbation * self._upper_bound /
            max(perturbation.norm(), self._upper_bound)
        )


class ConstraintFactory:

    def create_constraint(self, name, upper_bound=None):
        if name == 'none':
            return NoConstraint()
        elif name == 'linf':
            assert upper_bound is not None
            return LinfConstraint(upper_bound)
        elif name == 'l2':
            assert upper_bound is not None
            return L2Constraint(upper_bound)
