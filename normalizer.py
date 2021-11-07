class Normalizer():
    def min_max_normalize(self, lst):
        minimum = min(lst)
        maximum = max(lst)

        normalized = []
        for i in range(len(lst)):
            optim_value = (lst[i] - minimum) / (maximum - minimum)
            normalized.append(optim_value)
        return normalized
