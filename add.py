while True: 
    try:
        a = [[0 for j in range(3)] for i in range(2)]
        for i in range(2):
            a[i] = [int(x) for x in input().split()]
        b = [[0 for j in range(2)] for i in range(3)]
        for i in range(3):
            b[i] = [int(x) for x in input().split()]
        res = [[0 for j in range(2)] for i in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[i][j] += a[i][k] * b[k][j]

        for i in range(2):
            for j in range(2):
                print(res[i][j] , end = ' ')
            print()
    except:
        break
