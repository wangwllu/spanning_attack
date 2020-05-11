import os
import configparser
import argparse
from datetime import datetime

import torch

from subattack.utils.data import generate_1by1
from subattack.utils.data import get_loaders
from subattack.utils.data import idx_to_label
from subattack.utils.results import AttackResult
from subattack.utils.results import AttackDetailCollection

from subattack.strategies.svd import SvdBasisFetcherFactory
from subattack.strategies.adv_checkers import AdvCheckerFactory
from subattack.strategies.constraints import ConstraintFactory
from subattack.strategies.conventions import ConventionFactory
from subattack.strategies.gradient_estimators import GradientEstimatorFactory
from subattack.strategies.initializers import InitializerFactory
from subattack.strategies.loss import LossEvaluatorFactory
from subattack.strategies.models import ModelFactory
from subattack.strategies.samplers import SamplerFactory
from subattack.strategies.steepest import SteepestGradientTransformerFactory
from subattack.strategies.pools import PoolFetcherFactory

from subattack.attackers.gradient_attackers import GradientAttacker


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default=os.path.join('config', 'score.ini')
)
parser.add_argument(
    '--section',
    default='DEFAULT'
)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
params = config[args.section]

result_dir = os.path.join(
    params.get('result_dir'),
    'score',
    args.section,
    '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

with open(os.path.join(result_dir, 'config.ini'), 'w') as backup_configfile:
    config.write(backup_configfile)

torch.manual_seed(params.getint('seed'))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['TORCH_HOME'] = params.get('torch_home')
model = ModelFactory().create_model(
    params.get('model_name'), params.get('device')
)

manifold_loader, attack_loader = get_loaders(
    params.get('data_dir'),
    params.getint('size_manifold'),
    params.getint('size_attack'),
    params.getint('num_workers'),
    params.getint('batch_size')
)

adv_checker = AdvCheckerFactory().create_adv_checker(
    name='untargeted'
)

convention = ConventionFactory().create_convention(params.get('convention'))

constraint = ConstraintFactory().create_constraint(
    name=params.get('constraint'),
    upper_bound=params.getfloat('upper_bound')
)

steepest_gradient_transformer = SteepestGradientTransformerFactory(
).create_steepest_gradient_transformer(name=params.get('steepest'))

loss_evaluator = LossEvaluatorFactory().create_loss_evaluator(
    name=params.get('loss')
)

pool_fetcher = PoolFetcherFactory().create_pool_fetcher(
    name=params.get('pool'),
    loader=manifold_loader,
    max_size=params.getint('pool_size'),
    device=params.get('device')
)

pool = pool_fetcher.fetch()

basis_fetcher = SvdBasisFetcherFactory().create_svd_basis_fetcher(
    name=params.get('svd'),
    pool=pool,
    subspace_size=params.getint('subspace_size'),
    position=params.get('position'),
)
basis = basis_fetcher.fetch()

initializer = InitializerFactory().create_initializer(
    name=params.get('initializer'),
    radius=params.getfloat('radius'),
    shape=convention.shape,
    basis=basis,
    device=params.get('device')
)

sampler = SamplerFactory().create_sampler(
    name=params.get('sampler'),
    basis=basis,
    device=params.get('device')
)

gradient_estimator = GradientEstimatorFactory().create_gradient_estimator(
    name=params.get('gradient_estimator'),
    sampler=sampler,
    change=params.getfloat('change'),
    sample_size=params.getint('sample_size')
)

gradient_attacker = GradientAttacker(
    model, params.getfloat('step_size'), params.getint('budget'),
    adv_checker, convention, initializer, gradient_estimator,
    constraint, loss_evaluator, steepest_gradient_transformer
)

attack_detail_collection = AttackDetailCollection()

count = 0
for image, label in generate_1by1(attack_loader, params.get('device')):

    pred_label = model(image.unsqueeze(0)).squeeze().argmax()
    if pred_label != label:
        continue

    print('{}/{}'.format(count+1, params.getint('num_attack')))
    adv_image, adv_label, cost = gradient_attacker.solve(image, label)
    attack_result = AttackResult(
        image, label, adv_image, adv_label, cost, adv_checker
    )

    attack_result.save_image(count, idx_to_label, result_dir)
    attack_result.save_adv_image(count, idx_to_label, result_dir)

    attack_result.print_detail()

    attack_detail_collection.append(attack_result.detail)
    attack_detail_collection.save_detail(result_dir)

    count += 1
    if count >= params.getint('num_attack'):
        break

attack_detail_collection.print_summary()
attack_detail_collection.save_summary(result_dir)
