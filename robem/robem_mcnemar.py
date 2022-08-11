from statsmodels.stats.contingency_tables import mcnemar


def contingency_table(first_model: list, second_model: list):
    table = [[0, 0],
             [0, 0]]
    for y1, y2 in zip(first_model, second_model):
        table[int(not y1)][int(not y2)] += 1

    return table


def mcn_test(first_results, second_results, alpha=0.05):
    # calculate mcnemar test
    table = contingency_table(first_results, second_results)
    result = mcnemar(table, exact=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value

    if result.pvalue > alpha:
        # Classifiers have a similar proportion of errors on the test set: not significant
        print('Null hypothesis can not be rejected')
        print('There is not significant difference - p value :', result.pvalue)
        return False
    else:
        # Classifiers have a different proportion of errors on the test set: significant
        print('Null hypothesis rejected')
        print('There is significant difference - p value :', result.pvalue)
        return True
