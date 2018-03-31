
def LeaveOneOut(array):
  for i in range(len(array)):
    yield (array[:i] + array[i + 1:], array[i])


if __name__ == "__main__":
  v = [1, 2, 3, 4, 5]
  for train, test in LeaveOneOut(v):
    print(train, test)
  