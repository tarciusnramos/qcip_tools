#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from qcip_tools import derivatives_e

__version__ = '0.1'
__author__ = 'Tarcius N. Ramos'
__maintainer__ = 'Pierre Beaujean & Tarcius N. Ramos'
__email__ = 'pierre.beaujean@unamur.be & tarcius.nascimento@unamur.be'
__status__ = 'Development'

__longdoc__ = """
Try to fetch the dipole moment and the frequency dependant (hyper)polarizabilities from a csv file, then save the tensors, and
related quantities into another csv file. Rely on the availability of ``electrical_derivatives``.

"""


def get_arguments_parser(string=None):
    parser = argparse.ArgumentParser(description='Extract the LoProp of a selection of domains.')
    parser.add_argument('file', help='Files to be analysed.')
    parser.add_argument('-o', '--output_file', help='Output base name file. Default: electrical_derivatives.csv', 
                        default='electrical_derivatives.csv', type=str)
    
    return parser.parse_args(string)


def _columns_to_tensor(data):
    len_data = len(data)
    tensor = None

    if len_data == 3:
        tensor = np.empty((3), dtype=float)
    elif len_data == 9:
        tensor = np.empty((3, 3), dtype=float)
    elif len_data == 27:
        tensor = np.empty((3, 3, 3), dtype=float)
    elif len_data == 81:
        tensor = np.empty((3, 3, 3, 3), dtype=float)
    
    translate_coord = {'x': 0, 'y': 1, 'z': 2}

    for col in data.index:
        c = col.split('_')[1]
        tmp = [ translate_coord[x] for x in c ]
        tensor[tuple(tmp)] = data[col]
    
    return tensor


def append_data(data, properties):
    for k, v in properties.items():
        if k not in data.keys():
            data[k] = []
        data[k].append(v)
    
    return data


def main():
    args = get_arguments_parser()
    df = pd.read_csv(args.file)

    components = {
        'mu': [ f'mu_{i}' for i in 'x y z'.split() ],
        'alpha': [ f'alpha_{i}{j}' for i in 'x y z'.split() for j in 'x y z'.split() ],
        'beta': [f'beta_{i}{j}{k}' for i in 'x y z'.split() for j in 'x y z'.split() for k in 'x y z'.split() ],
        'gamma': [f'gamma_{i}{j}{k}{l}' for i in 'x y z'.split() for j in 'x y z'.split() for k in 'x y z'.split() for l in 'x y z'.split() ],
    }

    col_mu = [ c for c in df.columns if c in components['mu'] ]
    col_alpha = [ c for c in df.columns if c in components['alpha'] ]
    col_beta = [ c for c in df.columns if c in components['beta'] ]
    col_gamma = [ c for c in df.columns if c in components['gamma'] ]

    new_data = {}
    new_data['names'] = df['names']

    if len(col_mu) != 0:
        for _, row in df.iterrows():
            mu = derivatives_e.ElectricDipole()
            mu.components = _columns_to_tensor(row[col_mu])
            mu.compute_properties()
            new_data = append_data(new_data, mu.properties)
            components_dict = {}
            for c, v in zip(components['mu'], mu.flatten_components()):
                components_dict[c] = v
            new_data = append_data(new_data, components_dict)

    if len(col_alpha) != 0:
        for _, row in df.iterrows():
            alpha = derivatives_e.PolarisabilityTensor()
            alpha.components = _columns_to_tensor(row[col_alpha])
            alpha.compute_properties()
            new_data = append_data(new_data, alpha.properties)
            components_dict = {}
            for c, v in zip(components['alpha'], alpha.flatten_components()):
                components_dict[c] = v
            new_data = append_data(new_data, components_dict)

    if len(col_beta) != 0:
        for _, row in df.iterrows():
            beta = derivatives_e.FirstHyperpolarisabilityTensor()
            beta.components = _columns_to_tensor(row[col_beta])
            kw = {}
            if len(col_mu) != 0:
                kw = {'dipole' : row[col_mu].to_numpy()}
            beta.compute_properties(**kw)
            new_data = append_data(new_data, beta.properties)
            components_dict = {}
            for c, v in zip(components['beta'], beta.flatten_components()):
                components_dict[c] = v
            new_data = append_data(new_data, components_dict)

    if len(col_gamma) != 0:
        for _, row in df.iterrows():
            gamma = derivatives_e.SecondHyperpolarizabilityTensor()
            gamma.components = _columns_to_tensor(row[col_gamma])
            kw = {}
            if len(col_mu) != 0:
                kw = {'dipole' : row[col_mu].to_numpy()}
            gamma.compute_properties(**kw)
            new_data = append_data(new_data, gamma.properties)
            components_dict = {}
            for c, v in zip(components['gamma'], gamma.flatten_components()):
                components_dict[c] = v
            new_data = append_data(new_data, components_dict)
        

        
    new_data = pd.DataFrame(new_data)
    new_data.to_csv(args.output_file, index=False)

    


if __name__ == '__main__':
    main()
