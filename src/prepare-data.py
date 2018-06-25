import argparse
import glob
import json
import logging
import daiquiri
from tqdm import tqdm

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def merge(reports):
    output = []
    for i in tqdm(reports):
        logger.debug(f"Processing file {i}\n")
        with open(i, 'r') as f:
            content = json.load(f)
            for document in content:
                report = document['RisReport']
                output.append(report.replace('\n', ''))
    output = '\n'.join(output)
    return output


def write(output_file, output):
    with open(output_file, 'w') as o:
        o.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Prepare RIS data for topic modelling')
    parser.add_argument(
        '--ris-path',
        metavar="path",
        type=str,
        required=True,
        help="RIS file path, accepts * as wildcard")
    parser.add_argument('--output-file', default='output.txt')
    args = parser.parse_args()
    output_file = args.output_file

    reports = glob.glob(args.ris_path)
    logger.info('Running merger')
    output = merge(reports)
    write(output_file, output)
    logger.info('Merger Done')