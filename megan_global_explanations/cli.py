import os
import json
from typing import List

import click
from pycomex.cli import ExperimentCLI

from megan_global_explanations.utils import PATH
from megan_global_explanations.utils import get_version
from megan_global_explanations.utils import CsvString

cli = ExperimentCLI(
    name='exp',
    experiments_path=os.path.join(PATH, 'experiments'),
    version=get_version()
)

@click.command('anon', short_help='anonymizes all the contents of the repository')
@click.argument('path', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option('--exclude', type=CsvString(), default='json,jsonl,csv',
              help='comma separated list of file extensions to exclude from this process')
def anonymize(path: str, exclude: List[str]):
    identity_path = os.path.join(path, '.identity.json')

    assert os.path.exists(identity_path), (
        f'The file ".identity.json" which contains the anonymization mapping could not be found at the path '
        f'"{identity_path}". No de-anonymization cannot be performed.'
    )

    click.secho(f'anonymizing: {path}')
    click.secho(f'excluding extensions: {exclude}')

    with open(identity_path, mode='r') as file:
        identity_dict = json.load(file)

    counter = 0
    for root, folders, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # We actually can't be sure that files have file extensions (or that they just have a single
            # file extension) so we need catch an eventual exception, in which case we just assume that
            # there is not extension. In python this is actually more efficient than checking for all the
            # eventual special cases by hand
            try:
                name, extension = file_name.split('.')
            except ValueError:
                name, extension = file_name, ''

            # We absolutely do not want to apply this process to the identify file itself, as that would
            # completely corrupt the mapping that we still need to reverse this process.
            if file_name == '.identity.json' or extension in exclude:
                continue

            try:
                with open(file_path, mode='r') as file:
                    content = file.read()

                # Now inside the content we need to replace every mention of each of the words
                for identity_data in identity_dict.values():
                    content = content.replace(identity_data['real'], identity_data['anon'])

                with open(file_path, mode='w') as file:
                    file.write(content)
                    counter += 1
            except UnicodeDecodeError:
                continue

    click.secho(f'anonymized {counter} files')


@click.command('de-anon', short_help='de-anonymizes all the contents of the repository')
@click.argument('path', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option('--exclude', type=CsvString(), default='json,jsonl,csv',
              help='comma separated list of file extensions to exclude from this process')
def deanonymize(path: str, exclude: List[str]):
    identity_path = os.path.join(path, '.identity.json')

    assert os.path.exists(identity_path), (
        f'The file ".identity.json" which contains the anonymization mapping could not be found at the path '
        f'"{identity_path}". No de-anonymization cannot be performed.'
    )

    click.secho(f'de-anonymizing: {path}')
    click.secho(f'excluding extensions: {exclude}')

    with open(identity_path, mode='r') as file:
        identity_dict = json.load(file)

    counter = 0
    for root, folders, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # We actually can't be sure that files have file extensions (or that they just have a single
            # file extension) so we need catch an eventual exception, in which case we just assume that
            # there is not extension. In python this is actually more efficient than checking for all the
            # eventual special cases by hand
            try:
                name, extension = file_name.split('.')
            except ValueError:
                name, extension = file_name, ''

            # We absolutely do not want to apply this process to the identify file itself, as that would
            # completely corrupt the mapping that we still need to reverse this process.
            if file_name == '.identity.json' or extension in exclude:
                continue

            try:
                with open(file_path, mode='r') as file:
                    content = file.read()

                # Now inside the content we need to replace every mention of each of the words
                for identity_data in identity_dict.values():
                    content = content.replace(identity_data['anon'], identity_data['real'])

                with open(file_path, mode='w') as file:
                    file.write(content)
                    counter += 1
            except UnicodeDecodeError:
                continue

    click.secho(f'de-anonymized {counter} files')


cli.add_command(anonymize)
cli.add_command(deanonymize)


if __name__ == '__main__':
    cli()
