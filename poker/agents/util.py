
def remap_numbers(lst):
    # Replace all the cells with numbers so that the smallest number becomes 0,
    # the second smallest becomes 1 and so on. For example,  [None, 3, None 1] becomes
    # [None, 1, None, 0].
    nums = sorted(set(x for x in lst if x is not None))
    mapping = {num: i for i, num in enumerate(nums)}
    return [mapping[x] if x is not None else None for x in lst]