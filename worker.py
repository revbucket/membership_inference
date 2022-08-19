from fastargs.decorators import param
from fastargs import Param, Section, get_current_config
from fastargs.validation import And, OneOf
from fastargs.config import Config
from argparse import ArgumentParser
import types
from pathlib import Path
import numpy as np
import tqdm
from functools import cache
import importlib.util

INDEX_NONCE = -99999999
LOGDIR_NONCE = ''


Section('worker').params(
    index=Param(int, 'index of this job', default=INDEX_NONCE),
    main_import=Param(str, 'relativer python import module with main() to run',
                      required=True),
)




def collect_known_args(self, parser, disable_help=False):
    args, _ = parser.parse_known_args()
    for fname in args.config_file:
        self.collect_config_file(fname)

    args = vars(args)
    self.collect(args)
    self.collect_env_variables()


def make_config(quiet=False, conf_path=None):
    config = get_current_config()
    if conf_path is not None:
        config.collect_config_file(conf_path)

    f = types.MethodType(collect_known_args, config)
    config.collect_argparse_args = f

    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    if not quiet:
        config.summary()

    return config



@param('worker.main_import')
@param('worker.index')
def main(main_import, index):
    module = importlib.import_module(main_import)
    make_config(quiet=True)
    module.main(index=index)


if __name__ == '__main__':
    make_config(quiet=True)
    main()

