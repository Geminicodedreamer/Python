# This is the comparison function that will be passed to the sorting algorithm
def compare(point1, point2):
    # First, compare the x coordinates
    if point1[0] < point2[0]:
        return True
    elif point1[0] > point2[0]:
        return False

    # If the x coordinates are equal, compare the y coordinates
    if point1[1] < point2[1]:
        return True
    else:
        return False

# Read the number of test cases
num_cases = int(input())

# Loop through each test case
for i in range(num_cases):
    # Read the number of points
    num_points = int(input())

    # Create a list to store the points
    points = []

    # Read the coordinates of each point
    for j in range(num_points):
        x, y = map(int, input().split())
        points.append((x, y))

    # Sort the points using the comparison function
    points.sort(key=cmp_to_key(compare))

    # Print the test case number
    print("Test case {}:".format(i + 1))

    # Print the sorted points
    for point in points:
        print("{} {}".format(point[0], point[1]))