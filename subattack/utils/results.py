import os
import pandas as pd
from collections import OrderedDict
from torchvision.transforms.functional import to_pil_image

from subattack.strategies.adv_checkers import AdvChecker


class AttackResult:

    def __init__(self, image, label, adv_image, adv_label, cost,
                 adv_checker: AdvChecker):
        self._image = image
        self._label = label
        self._adv_image = adv_image
        self._adv_label = adv_label
        self._cost = cost
        self._adv_checker = adv_checker

    def successful(self):
        return self._adv_checker.successful(self._label, self._adv_label)

    @property
    def l2_perturbation(self):
        return (self._image - self._adv_image).norm().item()

    @property
    def linf_perturbation(self):
        return (self._image - self._adv_image).abs().max().item()

    def save_image(self, order, idx_to_label, result_dir):
        to_pil_image(self._image.cpu()).save(
            os.path.join(
                result_dir,
                '{:04d}_original_{}_{}.png'.format(
                    order, self._label.item(),
                    idx_to_label[self._label.item()]
                )
            )
        )

    def save_adv_image(self, order, idx_to_label, result_dir):
        to_pil_image(self._adv_image.cpu()).save(
            os.path.join(
                result_dir,
                '{:04d}_adversarial_{}_{}.png'.format(
                    order, self._adv_label.item(),
                    idx_to_label[self._adv_label.item()])
            )
        )

    @property
    def detail(self):
        return OrderedDict([
            ('label', self._label.item()),
            ('adv_label', self._adv_label.item()),
            ('successful', self.successful()),
            ('cost', self._cost),
            ('linf', self.linf_perturbation),
            ('l2', self.l2_perturbation)
        ])

    def print_detail(self):
        print('-' * 30)
        print('**detail**')
        print(pd.Series(self.detail))
        print('-' * 30)


class AttackDetailCollection:

    def __init__(self):
        self._detail_list = []

    def append(self, detail):
        self._detail_list.append(detail)

    def save_detail(self, result_dir):
        df = pd.DataFrame(self._detail_list)
        df.to_csv(os.path.join(result_dir, 'detail.csv'))

    @property
    def summary(self):
        df = pd.DataFrame(self._detail_list)
        mask = df['successful']
        success_rate = mask.sum() / df.shape[0]

        cost_series = df['cost'][mask]
        linf_series = df['linf'][mask]
        l2_series = df['l2'][mask]

        mean_cost = cost_series.mean()
        mean_linf = linf_series.mean()
        mean_l2 = l2_series.mean()

        median_cost = cost_series.median()
        median_linf = linf_series.median()
        median_l2 = l2_series.median()

        return OrderedDict([
            ('success rate', success_rate),
            ('mean cost', mean_cost),
            ('mean pixel linf', mean_linf),
            ('mean l2', mean_l2),
            ('median cost', median_cost),
            ('median pixel linf', median_linf),
            ('median l2', median_l2),
        ])

    def print_summary(self):
        print('-' * 30)
        print('**summary**')
        print(pd.Series(self.summary))
        print('-' * 30)

    def save_summary(self, result_dir):
        pd.Series(self.summary).to_csv(
            os.path.join(result_dir, 'summary.csv'), header=False
        )
