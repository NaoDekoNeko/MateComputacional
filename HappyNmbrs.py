col = {}

def isHappy(n):
    sum = 0
    for i in str(n):
        sum += int(i)**2
    if sum == 1:
        return True
    elif sum in col:
        return False
    else:
        col[sum] = False
        return isHappy(sum)

def happyNumbers(n):
    for i in range(1, n+1):
        if isHappy(i):
            col[i] = True
    return col

def main():
    print(isHappy(23))
    print("-------------------------")
    print(happyNumbers(100))

main()