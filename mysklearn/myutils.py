import numpy as np # use numpy's random number generation

# TODO: your reusable general-purpose functions here
def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def categorize_mpg(mpg_value):
    if mpg_value >= 45:
        return 10
    elif 37 <= mpg_value <= 44:
        return 9
    elif 31 <= mpg_value <= 36:
        return 8
    elif 27 <= mpg_value <= 30:
        return 7
    elif 24 <= mpg_value <= 26:
        return 6
    elif 20 <= mpg_value <= 23:
        return 5
    elif 17 <= mpg_value <= 19:
        return 4
    elif 15 <= mpg_value <= 16:
        return 3
    elif mpg_value == 14:
        return 2
    else:  # mpg_value <= 13
        return 1