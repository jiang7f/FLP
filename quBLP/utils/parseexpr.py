def split_expr(expr):
    ## split by '+' or '-'
    expr = expr.replace(' ','')
    terms = expr.split('+')
    all_terms = []
    for i in range(len(terms)):
        if '-' in terms[i]:
            new_terms = terms[i].split('-')
            all_terms.append(('+', new_terms[0]))
            for term in new_terms[1:]:
                if term == '':
                    continue
                all_terms.append(('-', term))
        else:
            all_terms.append(('+', terms[i]))
    return all_terms