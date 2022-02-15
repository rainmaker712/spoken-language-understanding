import argparse

def get_slu_args():
    
    parser = argparse.ArgumentParser(description="spoken-langauge-understanding parser", allow_abbrev=False)
    parser = _add_data_parser(parser)
    parser = _add_text_parser(parser)
    parser = _add_slu_parser(parser)

    args, _ = parser.parse_known_args()

    return args

def _add_data_parser(parser):
    group = parser.add_argument_group(title="Args for dataloader")
    group.add_argument("--data-path", type=str, default="./fluent_speech_commands_dataset")
    group.add_argument("--batch-size", type=int, default=2)
    group.add_argument("--num-workers", type=int, default=0)

    return parser

def _add_text_parser(parser):
    group = parser.add_argument_group(title="Args for text model")
    group.add_argument("--text-model-name", type=str, default="roberta-base")
    return parser

def _add_slu_parser(parser):
    group = parser.add_argument_group(title="Args for slu model")
    group.add_argument("--num-intents", type=int, default=31)
    group.add_argument("--epochs", type=int, default=2)
    group.add_argument("--fp16", type=bool, default=False)
    group.add_argument("--use-scheduler", type=bool, default=True)
    group.add_argument("--learning-rate", type=float, default=2e-5)
    group.add_argument("--warmup-proportion", type=float, default=0.01)
    group.add_argument("--weight-decay", type=float, default=0.01)
    group.add_argument("--feature-size", type=int, default=80)
    group.add_argument("--output-embedding-size", type=int, default=512)
    group.add_argument("-num-workers", type=int, default=0)

    return parser