from robem_main_single import main

if __name__ == "__main__":
    datasets = [
        #'itunes-amazon',
        #'abt-buy',
        #'amazon-google',
        #'beer-rates',
        #'dblp-acm',
        #'dblp-scholar',
        #'fodors-zagats',
        #'walmart-amazon',
        #'dirty-itunes-amazon',
        #'dirty-dblp-acm',
        #'dirty-dblp-scholar',
        #'dirty-walmart-amazon',
        #'company'
    ]

    results = {}
    for ds in datasets:
        f1 = main(ds, rob=True, aug_size=1)
        results[ds] = f1

    print(results)
