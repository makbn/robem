import json


def get_dataset_name(name):
    datasets = {'amazon-google':'Structured/Amazon-Google',
                'beer-rates':'Structured/Beer',
                'dblp-acm':'Structured/DBLP-ACM',
                'dblp-scholar': 'Structured/DBLP-GoogleScholar',
                'fodors-zagats':'Structured/Fodors-Zagats',
                'itunes-amazon':'Structured/iTunes-Amazon',
                'walmart-amazon':'Structured/Walmart-Amazon',
                'abt-buy':'Textual/Abt-Buy',
                'company': 'Textual/Company'
                }
    return datasets[name]

