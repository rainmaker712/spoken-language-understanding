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

    return parser

def _add_text_parser(parser):
    group = parser.add_argument_group(title="Args for text model")
    group.add_argument("--text-model-name", type=str, default="roberta-base")
    return parser

def _add_slu_parser(parser):
    group = parser.add_argument_group(title="Args for slu model")
    group.add_argument("--num_intents", type=int, default=30)
    group.add_argument("--epochs", type=int, default=2)
    group.add_argument("--fp16", type=bool, default=False)
    group.add_argument("--use_scheduler", type=bool, default=True)
    group.add_argument("--learning_rate", type=float, default=2e-5)
    group.add_argument("--warmup_proportion", type=float, default=0.01)
    group.add_argument("--weight_decay", type=float, default=0.01)

    return parser