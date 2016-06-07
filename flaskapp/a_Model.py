## ======================================
def ModelIt(fromUser='Default', births=[]):
    """
     A function to create a very simple model
    """

    in_month = len(births)
    print 'The number born is {0}'.format(in_month)
    result = in_month
    if fromUser != 'Default':
        return result
    else:
        return 'check your input'