import argparse
from typing import Any, Iterable

from accelerate.utils import release_memory


def parse_args() -> argparse.Namespace:
    """Parse CLI args for launching train/test runners."""
    parser = argparse.ArgumentParser(description='Task')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--runners', type=str, default=None, help='runners of executing task')
    args = parser.parse_args()
    return args


def run_tasks(config: str | Any, runners: str | Iterable[str] | None = None) -> None:
    from giga_train import Tester, Trainer, load_config, utils

    config = load_config(config)
    if runners is None:
        runners = config.runners
    if isinstance(runners, str):
        runners = runners.split(',')
    if not isinstance(runners, (list, tuple)):
        runners = [runners]
    for runner in runners:
        runner = utils.import_function(runner)
        runner = runner.load(config)
        runner.print(config)
        if isinstance(runner, Trainer):
            runner.save_config(config)
            if config.train.get('resume', False):
                runner.resume()
            runner.train()
        elif isinstance(runner, Tester):
            runner.test()
        else:
            assert False
        release_memory()


def main() -> None:
    args = parse_args()
    run_tasks(args.config, args.runners)


if __name__ == '__main__':
    main()
