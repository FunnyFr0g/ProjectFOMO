
n = int(input().strip())
tanks = []
for _ in range(n):
    data = input().split()
    l = int(data[0])
    r = int(data[1])
    a = int(data[2])
    tanks.append((l, r, a))

best = 0
for i in range(n):
    total = tanks[i][2]
    current = tanks[i][0]
    j = i - 1
    while j >= 0 and current > 0:
        add = min(current, tanks[j][2])
        total += add
        current = min(tanks[j][0], current - add)
        j -= 1

    current = tanks[i][1]
    j = i + 1
    while j < n and current > 0:
        add = min(current, tanks[j][2])
        total += add
        current = min(tanks[j][1], current - add)
        j += 1

    if total > best:
        best = total
print(best)


