from abc import ABC, abstractmethod


class AdvChecker(ABC):

    @abstractmethod
    def successful(self, label, pred_label):
        pass


class UntargetedAdvChecker(AdvChecker):

    def successful(self, label, pred_label):
        return bool((label != pred_label).item())


class TargetedAdvChecker(AdvChecker):

    def successful(self, label, pred_label):
        return bool((label == pred_label).item())


class AdvCheckerFactory:

    def create_adv_checker(self, name):
        if name == 'untargeted':
            return UntargetedAdvChecker()
        elif name == 'targeted':
            return TargetedAdvChecker()
        else:
            raise Exception('unsupported adv checker')
